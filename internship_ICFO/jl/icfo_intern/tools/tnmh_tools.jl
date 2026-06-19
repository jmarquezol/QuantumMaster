# tools/tnmh_tools.jl
# =====================================================================
# Reusable primitives for the TNMH mixing-time study (RESEARCH_PLAN.md).
# Assumes `include("main.jl")` has already been run (provides
# create_ising_peps, compute_bottom_envs, sample_classical_1d,
# sample_config_opt, measure_energy).
#
# Naming honours CLAUDE.md §2a: `C` = full-lattice Mengersen-Tweedie
# constant `max_x π/q`; `ρ` = per-layer marginal ratio. The functions
# here are the FIRST in-repo implementation of the true `C` machinery
# (deterministic log q for a pinned config + full enumeration).
# =====================================================================

using LinearAlgebra, Random, Statistics

set_seed(s::Integer) = Random.seed!(s)

# numerically stable log Σ exp
function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    isfinite(m) || return m
    return m + log(sum(exp.(v .- m)))
end

# ---------------------------------------------------------------------
# Phase 0.1 — deterministic proposal log-probability  log q(config)
# ---------------------------------------------------------------------

# Pinned twin of sample_classical_1d (main.jl:95): identical conditional
# machinery, but the spin at each site is taken from `pinned_spins`
# (0/1) instead of drawn. Returns Σ log p_chosen = log q(row | env).
function sample_classical_1d_pinned(row_mps::MPS, s_inds, pinned_spins)
    Ly = length(row_mps)
    log_prob_row = 0.0

    # right environments (right -> left), normalised for stability
    R = Vector{ITensor}(undef, Ly)
    temp = ITensor(1.0)
    for y in Ly:-1:1
        T_traced = row_mps[y] * ITensor([1.0, 1.0], s_inds[y])
        temp *= T_traced
        n = norm(temp); if n > 0; temp ./= n; end
        R[y] = temp
    end

    L_env = ITensor(1.0)
    for y in 1:Ly
        T = row_mps[y]
        proj0 = ITensor([1.0, 0.0], s_inds[y])
        proj1 = ITensor([0.0, 1.0], s_inds[y])
        w0 = max(0.0, scalar(L_env * (T * proj0) * (y < Ly ? R[y+1] : ITensor(1.0))))
        w1 = max(0.0, scalar(L_env * (T * proj1) * (y < Ly ? R[y+1] : ITensor(1.0))))
        total_w = w0 + w1
        p0 = total_w < 1e-15 ? 0.5 : w0 / total_w

        chosen_spin = pinned_spins[y]
        p_chosen = chosen_spin == 0 ? p0 : (1.0 - p0)
        log_prob_row += p_chosen > 1e-15 ? log(p_chosen) : -Inf

        chosen_proj = chosen_spin == 0 ? proj0 : proj1
        L_env *= (T * chosen_proj)
        n = norm(L_env); if n > 0; L_env ./= n; end
    end
    return log_prob_row
end

# Build PEPS + bottom environments once (so enumeration reuses them).
function build_peps_envs(Lx::Int, Ly::Int, beta::Float64, D_bound::Int, J::Float64=1.0)
    A, s_inds, v_inds = create_ising_peps(Lx, Ly, beta, J)
    bottom_envs = compute_bottom_envs(A, s_inds, v_inds, Lx, Ly, D_bound)
    return A, s_inds, v_inds, bottom_envs
end

# Low-level: deterministic log q(config) given precomputed PEPS+envs.
# Mirrors sample_config_opt (main.jl:147) exactly, pinning each row.
function proposal_logprob!(config, A, s_inds, v_inds, bottom_envs, Lx::Int, Ly::Int, D_bound::Int)
    top_env = nothing
    b = [Index(2, "Site,b=$y") for y in 1:Ly]
    log_prob_tot = 0.0

    for x in 1:Lx
        eff_tensors = ITensor[]
        for y in 1:Ly
            T = A[x, y]
            if top_env !== nothing
                T_top = replaceinds(top_env[y], b[y] => v_inds[x-1, y])
                T *= T_top
            end
            if x < Lx
                T *= bottom_envs[x+1][y]
            end
            push!(eff_tensors, T)
        end
        row_mps = MPS(eff_tensors)

        log_prob_tot += sample_classical_1d_pinned(row_mps, s_inds[x, :], config[x, :])

        # fix the row's spins into an MPO and fold into top_env
        mpo_tensors = ITensor[]
        for y in 1:Ly
            proj = ITensor(config[x, y] == 0 ? [1.0, 0.0] : [0.0, 1.0], s_inds[x, y])
            push!(mpo_tensors, A[x, y] * proj)
        end

        if top_env === nothing
            top_tensors = ITensor[]
            for y in 1:Ly
                push!(top_tensors, replaceinds(mpo_tensors[y], [v_inds[1, y]] => [b[y]]))
            end
            top_env = MPS(top_tensors)
            normalize!(top_env)
        elseif x < Lx
            mpo_tensors_replaced = ITensor[]
            for y in 1:Ly
                T = replaceinds(mpo_tensors[y], [v_inds[x-1, y], v_inds[x, y]] => [b[y], b[y]'])
                push!(mpo_tensors_replaced, T)
            end
            top_env = noprime(apply(MPO(mpo_tensors_replaced), top_env; maxdim=D_bound, cutoff=1e-10))
            normalize!(top_env)
        end
    end
    return log_prob_tot
end

# Convenience wrapper (builds its own PEPS+envs).
function proposal_logprob(config, Lx::Int, Ly::Int, beta::Float64, D_bound::Int, J::Float64=1.0)
    A, s_inds, v_inds, bottom_envs = build_peps_envs(Lx, Ly, beta, D_bound, J)
    return proposal_logprob!(config, A, s_inds, v_inds, bottom_envs, Lx, Ly, D_bound)
end

# ---------------------------------------------------------------------
# Phase 0.3 — full enumeration harness  (true C, π, q, w on small lattices)
# ---------------------------------------------------------------------

# integer -> (Lx×Ly) 0/1 config, column-major bit ordering
function config_from_int(i::Integer, Lx::Int, Ly::Int)
    config = zeros(Int, Lx, Ly)
    for k in 1:(Lx * Ly)
        x = ((k - 1) % Lx) + 1
        y = ((k - 1) ÷ Lx) + 1
        config[x, y] = (i >> (k - 1)) & 1
    end
    return config
end

# Enumerate all 2^(Lx·Ly) configs. Z from exact energies (brute force).
# Returns a NamedTuple with logq, logpi, logw, E, w, C (=max w), the
# proposal-normalisation check sum_q (≈1 iff log q is a valid pmf),
# logZ, and the index `argmax` of the worst-case (C-realising) config.
function enumerate_weights(Lx::Int, Ly::Int, beta::Float64, D_bound::Int; J::Float64=1.0)
    N = Lx * Ly
    N <= 22 || error("enumerate_weights: 2^$N too large; keep Lx·Ly ≲ 18–20")
    M = 1 << N
    A, s_inds, v_inds, bottom_envs = build_peps_envs(Lx, Ly, beta, D_bound, J)

    logq = zeros(Float64, M)
    E = zeros(Float64, M)
    for i in 0:(M - 1)
        cfg = config_from_int(i, Lx, Ly)
        logq[i+1] = proposal_logprob!(cfg, A, s_inds, v_inds, bottom_envs, Lx, Ly, D_bound)
        E[i+1] = measure_energy(cfg, J)
    end

    logpi_un = -beta .* E
    logZ = logsumexp(logpi_un)
    logpi = logpi_un .- logZ
    logw = logpi .- logq
    w = exp.(logw)
    return (logq=logq, logpi=logpi, logw=logw, E=E, w=w,
            C=maximum(w), sum_q=sum(exp.(logq)), logZ=logZ, argmax=argmax(w))
end

# ---------------------------------------------------------------------
# Phase 1 — IMH transition kernel + spectrum
# ---------------------------------------------------------------------

# Dense IMH kernel on the 2^N state space:
#   P[a,b] = q(b)·min(1, w(b)/w(a))   (b≠a),   P[a,a] = 1 − Σ_{b≠a}.
function imh_kernel(logq::AbstractVector, logw::AbstractVector)
    M = length(logq)
    M <= 4096 || @warn "imh_kernel: dense $M×$M matrix is large ($(round(M^2*8/1e9,digits=2)) GB)"
    q = exp.(logq)
    P = zeros(Float64, M, M)
    @inbounds for a in 1:M
        s = 0.0
        for b in 1:M
            if a != b
                P[a, b] = q[b] * min(1.0, exp(logw[b] - logw[a]))
                s += P[a, b]
            end
        end
        P[a, a] = 1.0 - s
    end
    return P
end

# Second-largest eigenvalue of P (Liu 1996: should equal 1 − 1/C).
function lambda2(P::AbstractMatrix)
    ev = sort(real.(eigvals(P)), rev=true)
    return ev[2]
end

# TV distance ‖p − π‖_TV = ½ Σ|p−π|
tv_distance(p::AbstractVector, pivec::AbstractVector) = 0.5 * sum(abs.(p .- pivec))

# Iterate P^n from a worst-case start; return ‖P^n(x0,·) − π‖_TV vs n.
function tv_decay(P::AbstractMatrix, pivec::AbstractVector, x0::Int, nmax::Int)
    M = size(P, 1)
    p = zeros(M); p[x0] = 1.0
    out = zeros(Float64, nmax)
    for n in 1:nmax
        p = vec(p' * P)          # row-vector update (left action)
        out[n] = tv_distance(p, pivec)
    end
    return out
end

# ---------------------------------------------------------------------
# Phase 2 — ρ(Ly,D,β): per-layer marginal ratio  (CLAUDE.md §11.1 relabel)
# ---------------------------------------------------------------------
# Clean, correctly-named reimplementation of the notebook's
# `compute_asymptotic_C` (cell 19). The returned number is ρ, NOT C.
# `simulation.ipynb` is left untouched as historical record.

function compute_rho_plateau(Lx::Int, Ly::Int, beta::Float64, D_bound::Int; J::Float64=1.0)
    A, s_inds, v_inds = create_ising_peps(Lx, Ly, beta, J)
    b = [Index(2, "Site,b=$y") for y in 1:Ly]

    tensors_Lx = ITensor[]
    for y in 1:Ly
        T = A[Lx, y] * ITensor([1.0, 1.0], s_inds[Lx, y])
        T = replaceinds(T, [v_inds[Lx-1, y]] => [b[y]])
        push!(tensors_Lx, T)
    end
    E_exact = MPS(tensors_Lx)
    E_trunc = MPS(tensors_Lx)
    logZe = 0.0; logZt = 0.0
    exact_maxdim = 2^Ly

    for x in Lx-1:-1:2
        tensors_mpo = ITensor[]
        for y in 1:Ly
            T = A[x, y] * ITensor([1.0, 1.0], s_inds[x, y])
            T = replaceinds(T, [v_inds[x, y], v_inds[x-1, y]] => [b[y], b[y]'])
            push!(tensors_mpo, T)
        end
        MPOx = MPO(tensors_mpo)

        E_exact = noprime(apply(MPOx, E_exact; maxdim=exact_maxdim, cutoff=1e-15))
        orthogonalize!(E_exact, 1); ne = norm(E_exact[1]); E_exact[1] ./= ne; logZe += log(ne)

        E_trunc = noprime(apply(MPOx, E_trunc; maxdim=D_bound, cutoff=1e-10))
        orthogonalize!(E_trunc, 1); nt = norm(E_trunc[1]); E_trunc[1] ./= nt; logZt += log(nt)
    end

    max_ratio = 0.0
    x = 2  # measure the genuine top row (row 1): no upper bond, no [1,1] cap
    for i in 0:(2^Ly - 1)
        spins = [(i >> (j - 1)) & 1 for j in 1:Ly]
        rte = ITensor[]; rtt = ITensor[]
        for y in 1:Ly
            T = A[x-1, y]
            sv = spins[y] == 0 ? [1.0, 0.0] : [0.0, 1.0]
            T *= ITensor(sv, s_inds[x-1, y])
            push!(rte, T * replaceinds(E_exact[y], b[y] => v_inds[x-1, y]))
            push!(rtt, T * replaceinds(E_trunc[y], b[y] => v_inds[x-1, y]))
        end
        pe = abs(scalar(reduce(*, rte)))
        pt = abs(scalar(reduce(*, rtt)))
        if pt > 1e-30 && pe > 1e-30
            r = exp((log(pe) + logZe) - (log(pt) + logZt))
            r > max_ratio && (max_ratio = r)
        end
    end
    return max_ratio
end

# Per-row version: ρ_x and fidelity vs "rows from bottom" (relabel of
# run_error_tracker, cell 9). worst_case_ratios -> rho_x; the upper bond
# of an interior measurement row is [1,1]-capped (marginal — see §9.2).
function rho_tracker(Lx::Int, beta::Float64, D_bound::Int; Ly::Int=8, J::Float64=1.0)
    A, s_inds, v_inds = create_ising_peps(Lx, Ly, beta, J)
    b = [Index(2, "Site,b=$y") for y in 1:Ly]

    tensors_Lx = ITensor[]
    for y in 1:Ly
        T = A[Lx, y] * ITensor([1.0, 1.0], s_inds[Lx, y])
        T = replaceinds(T, [v_inds[Lx-1, y]] => [b[y]])
        push!(tensors_Lx, T)
    end
    E_exact = MPS(tensors_Lx); E_trunc = MPS(tensors_Lx)
    logZe = 0.0; logZt = 0.0
    exact_maxdim = 2^Ly

    rows = Int[]; fidelity = Float64[]; rho_x = Float64[]
    for x in Lx-1:-1:2
        tensors_mpo = ITensor[]
        for y in 1:Ly
            T = A[x, y] * ITensor([1.0, 1.0], s_inds[x, y])
            T = replaceinds(T, [v_inds[x, y], v_inds[x-1, y]] => [b[y], b[y]'])
            push!(tensors_mpo, T)
        end
        MPOx = MPO(tensors_mpo)

        E_exact = noprime(apply(MPOx, E_exact; maxdim=exact_maxdim, cutoff=1e-16))
        orthogonalize!(E_exact, 1); ne = norm(E_exact[1]); E_exact[1] ./= ne; logZe += log(ne)
        E_trunc = noprime(apply(MPOx, E_trunc; maxdim=D_bound, cutoff=1e-10))
        orthogonalize!(E_trunc, 1); nt = norm(E_trunc[1]); E_trunc[1] ./= nt; logZt += log(nt)

        push!(fidelity, abs(inner(E_exact, E_trunc)))

        mr = 0.0
        for i in 0:(2^Ly - 1)
            spins = [(i >> (j - 1)) & 1 for j in 1:Ly]
            rte = ITensor[]; rtt = ITensor[]
            for y in 1:Ly
                T = A[x-1, y]
                sv = spins[y] == 0 ? [1.0, 0.0] : [0.0, 1.0]
                T *= ITensor(sv, s_inds[x-1, y])
                if (x - 1) > 1
                    T *= ITensor([1.0, 1.0], v_inds[x-2, y])   # [1,1] marginal cap
                end
                push!(rte, T * replaceinds(E_exact[y], b[y] => v_inds[x-1, y]))
                push!(rtt, T * replaceinds(E_trunc[y], b[y] => v_inds[x-1, y]))
            end
            pe = abs(scalar(reduce(*, rte)))
            pt = abs(scalar(reduce(*, rtt)))
            if pt > 1e-30 && pe > 1e-30
                r = exp((log(pe) + logZe) - (log(pt) + logZt))
                r > mr && (mr = r)
            end
        end
        push!(rows, Lx - x + 1)
        push!(rho_x, mr)
    end
    return rows, fidelity, rho_x
end

# Saturated boundary MPS (rows 2..Lx contracted, fixed point for large Lx).
saturated_boundary_mps(Lx::Int, Ly::Int, beta::Float64, D_bound::Int; J::Float64=1.0) =
    compute_bottom_envs(create_ising_peps(Lx, Ly, beta, J)..., Lx, Ly, D_bound)[2]

# Phase 2.4 — discarded-weight proxy ε_D: 1 − |⟨e|e_D⟩|² for the exact (e)
# vs D-truncated (e_D) saturated boundary MPS (robust: uses only inner/norm).
function truncation_error_fidelity(Lx::Int, Ly::Int, beta::Float64, D::Int; J::Float64=1.0)
    e  = saturated_boundary_mps(Lx, Ly, beta, 2^Ly; J=J)
    eD = saturated_boundary_mps(Lx, Ly, beta, D;     J=J)
    ov = abs(inner(e, eD)) / (norm(e) * norm(eD))
    return sqrt(max(0.0, 1 - ov^2))
end

# ---------------------------------------------------------------------
# Phase 3 — average-case / typical mixing diagnostics
# ---------------------------------------------------------------------

# Draw n configs from q; self-normalised log-weights (E_q[w]=1 fixes scale,
# so logZ need not be known a priori — estimated from the sample).
function weight_samples(Lx::Int, Ly::Int, beta::Float64, D_bound::Int, n::Int; J::Float64=1.0)
    E = zeros(Float64, n); logq = zeros(Float64, n)
    for i in 1:n
        cfg, lq = sample_config_opt(Lx, Ly, beta, D_bound, J)
        E[i] = measure_energy(cfg, J)
        logq[i] = lq
    end
    logZ_est = logsumexp(-beta .* E .- logq) - log(n)   # Z = E_q[e^{-βE}/q]
    logw = (-beta .* E .- logZ_est) .- logq
    return (E=E, logq=logq, logw=logw, logZ_est=logZ_est)
end

# χ²(π‖q) = E_q[w²] − 1 = Var_q(w), exact from enumerate_weights output.
chi2_exact(res) = sum(exp.(res.logq .+ 2 .* res.logw)) - 1.0

# χ² by sampling, with reliability diagnostics (ESS, largest w).
function chi2_sampled(logw::AbstractVector)
    w = exp.(logw)
    return (chi2 = mean(w .^ 2) - 1.0,
            ess  = sum(w)^2 / sum(w .^ 2),
            wmax = maximum(w),
            n    = length(w))
end

# Restricted / typical-set constant C_S = max_{x∈S} w(x).
restricted_C(logw::AbstractVector, mask::AbstractVector{Bool}) =
    any(mask) ? exp(maximum(logw[mask])) : NaN

# Windowed integrated autocorrelation time (Sokal). Units = input steps.
function integrated_autocorr(series::AbstractVector; c::Float64=5.0)
    x = series .- mean(series)
    n = length(x)
    v0 = sum(x .^ 2) / n
    v0 == 0 && return 0.5
    tau = 0.5
    for t in 1:(n - 1)
        ct = (sum(@views x[1:n-t] .* x[t+1:n]) / (n - t)) / v0
        tau += ct
        t > c * tau && break
    end
    return tau
end

# ---------------------------------------------------------------------
# Phase 4 — independent validations
# ---------------------------------------------------------------------

# IMH CFTP coalescence (Corcoran–Tweedie): all coupled chains coalesce the
# first step the worst-case config s★ (argmax w) is proposed. Mean
# coalescence time = 1/q(s★) = C/π(s★). Returns empirical vs predicted.
function cftp_coalescence(Lx::Int, Ly::Int, beta::Float64, D_bound::Int;
                          ntrials::Int=200, maxsteps::Int=10^6, J::Float64=1.0)
    res = enumerate_weights(Lx, Ly, beta, D_bound; J=J)
    star_cfg = config_from_int(res.argmax - 1, Lx, Ly)
    times = Int[]
    for _ in 1:ntrials
        steps = 0
        while steps < maxsteps
            steps += 1
            cfg, _ = sample_config_opt(Lx, Ly, beta, D_bound, J)
            cfg == star_cfg && break
        end
        push!(times, steps)
    end
    qstar = exp(res.logq[res.argmax])
    return (mean_time = mean(times), predicted = 1 / qstar, C = res.C, times = times)
end

# Single-spin-flip Metropolis (Glauber) reference. One "sample" = one full
# lattice sweep of Lx·Ly attempted flips. Returns τ_int of energy (sweeps).
function glauber_tau_int(Lx::Int, Ly::Int, beta::Float64, N_samples::Int;
                         J::Float64=1.0, burnin_frac::Float64=0.2)
    spins = rand((-1, 1), Lx, Ly)
    energy(s) = -J * (sum(s[:, 1:end-1] .* s[:, 2:end]) + sum(s[1:end-1, :] .* s[2:end, :]))
    function dE(s, x, y)
        nb = 0
        x > 1  && (nb += s[x-1, y]); x < Lx && (nb += s[x+1, y])
        y > 1  && (nb += s[x, y-1]); y < Ly && (nb += s[x, y+1])
        return 2 * J * s[x, y] * nb
    end
    Es = Float64[]
    for _ in 1:N_samples
        for _ in 1:(Lx * Ly)
            x = rand(1:Lx); y = rand(1:Ly)
            d = dE(spins, x, y)
            (d <= 0 || rand() < exp(-beta * d)) && (spins[x, y] *= -1)
        end
        push!(Es, energy(spins))
    end
    burn = Int(floor(burnin_frac * N_samples))
    return (tau_int = integrated_autocorr(Es[burn+1:end]), E = Es)
end
