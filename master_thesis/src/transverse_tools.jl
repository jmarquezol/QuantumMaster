# ──────────────────────────────────────────────────────────────────────────────
# Transverse-contraction toolkit: build the temporal MPO, run the (block) power
# method, extract generalized temporal entropies, and the TDVP Schrödinger
# cross-check. Model-agnostic where it helps; the Alcaraz path is the default.
# ──────────────────────────────────────────────────────────────────────────────

overlap_lr(L::MPS, R::MPS) = overlap_noconj(L, R)

"""
    build_alcaraz_tmpo(target_T; p, lambda, dt, nbeta, MPO_alg) → (mpo, scaffold)

Returns the rotated forward tMPO (the spatial transfer matrix) and a structured
seed tMPS on the same time-site indices. The scaffold is symmetry-special and gets
overwritten with random complex tensors before any power method.
"""
function build_alcaraz_tmpo(target_T::Float64;
        p::Float64=0.1, lambda::Float64=1.0, dt::Float64=0.1,
        nbeta::Int=0, MPO_alg::String="VD2")
    Ntime_steps = round(Int, target_T / dt)
    Nsteps      = Ntime_steps + nbeta
    s           = Index(2, "S=1/2")
    init_state  = complex(state(s, "X+"))
    RECIPES     = Dict("WI"=>AlcarazWI(), 
                        "WII"=>AlcarazWII(), 
                        "VD2"=>AlcarazVD2())
    mp          = AlcarazParams(lambda=lambda, p=p, phys_site=s)
    tp          = tMPOParams(mp=mp, dt=dt, nbeta=nbeta, scheme=RECIPES[MPO_alg],
                             dbeta=-im*dt, bl=init_state)
    b           = FwtMPOBlocks(tp)
    spatial_bond_dim = dim(inds(b.Wc, "Site,time")[1])
    time_sites  = addtags(siteinds(spatial_bond_dim, Nsteps; conserve_qns=false), "time")
    mpo         = fw_tMPO(b, time_sites, tr=init_state)
    scaffold    = fw_tMPS(b, time_sites; tr=init_state, LR=:right)
    return mpo, scaffold
end

# truncated linear combination of MPS:  Σ_i coeffs[i] * vecs[i]
#   exact "directsum" combine then SVD-truncate
function lincomb_mps(coeffs::AbstractVector, vecs::AbstractVector{MPS};
                     cutoff::Float64=1e-12, maxdim::Int=256)
    acc = coeffs[1] * vecs[1]
    for i in 2:lastindex(vecs)
        acc = +(acc, coeffs[i] * vecs[i]; alg="directsum")
    end
    truncate!(acc; cutoff=cutoff, maxdim=maxdim)
    return acc
end

"""
    block_transfer_eigs(mpo, scaffold; k, maxdim, cutoff, itermax, eps_conv,
                        n_track, cond_thresh, maxdims) → (theta, L, R, info)

Block (subspace) power method for the leading `k` eigenvalues of the non-Hermitian
transfer operator `mpo`, with separate left/right bases and a fully non-conjugating
(overlap_noconj) oblique Rayleigh-Ritz. Robust through the even-sector degeneracy
where single-vector deflation diverges.

`maxdims` is an OPTIONAL bond-dim ramp: when given a vector, iteration `it` truncates to
`maxdims[min(it,end)]` (cheap early iterations, then grow to the cap); when `nothing`
(default) the fixed `maxdim` is used at every iteration — i.e. exactly the original behavior.

`seedL`/`seedR` are OPTIONAL warm starts: vectors of MPS on `siteinds(scaffold)` used as the
initial left/right bases (any shorter than `k` is padded with random vectors). Defaults `nothing`
⇒ fully random seeds (original behavior). A converged pair reused here converges in a few iters.

info keys: :niters, :reason, :condS (final), :condS_hist, :dtheta, :theta, :theta_eigen.
"""
function block_transfer_eigs(mpo::MPO, scaffold::MPS;
        k::Int=4, maxdim::Int=256, cutoff::Float64=1e-12,
        itermax::Int=300, eps_conv::Float64=1e-8, n_track::Int=2,
        cond_thresh::Float64=1e10,
        maxdims::Union{Nothing,AbstractVector{<:Integer}}=nothing,
        seedL::Union{Nothing,AbstractVector{MPS}}=nothing,
        seedR::Union{Nothing,AbstractVector{MPS}}=nothing)

    sit  = siteinds(scaffold)
    mpoT = swapprime(mpo, 0, 1)   # pure transpose (NO conjugation): ⟨L|mpoᵀ|R⟩ = ⟨mpo L|R⟩
    md_at(it) = maxdims === nothing ? maxdim : Int(maxdims[min(it, length(maxdims))])

    rand_mps() = normalize(complex.(randomMPS(sit, linkdims=2k)))
    seed_block(s) = s === nothing ? MPS[rand_mps() for _ in 1:k] :
        MPS[i <= length(s) ? normalize(complex.(s[i])) : rand_mps() for i in 1:k]
    R = seed_block(seedR)
    L = seed_block(seedL)

    theta       = fill(NaN + 0im, k)
    theta_prev  = fill(NaN + 0im, k)
    dtheta_hist = Float64[]
    condS_hist  = Float64[]
    condS_last  = NaN
    reason      = "maxiter"
    niters      = 0

    for it in 1:itermax
        niters = it
        md = md_at(it)                                  # per-iteration bond-dim cap (ramp or fixed)
        AR  = MPS[applyn(mpo,  R[j]; cutoff=cutoff, maxdim=md) for j in 1:k]
        ATL = MPS[applyn(mpoT, L[j]; cutoff=cutoff, maxdim=md) for j in 1:k]

        # oblique Rayleigh-Ritz pencil (non-conjugating)
        S = Matrix{ComplexF64}(undef, k, k)
        M = Matrix{ComplexF64}(undef, k, k)
        for i in 1:k, j in 1:k
            S[i, j] = overlap_noconj(L[i], R[j])
            M[i, j] = overlap_noconj(L[i], AR[j])
        end
        condS_last = cond(S)
        push!(condS_hist, condS_last)

        # right Ritz: solve S⁻¹M v = λ v via pinv (NOT eigen(M,S): Inf on near-singular S)
        W  = pinv(S; rtol=1e-12) * M
        Fr = eigen(W)
        permr = sortperm(abs.(Fr.values); rev=true)
        theta = Fr.values[permr]
        VR    = Fr.vectors[:, permr]

        # left Ritz from the transposed pencil, matched to θ by NEAREST complex value
        Wl = pinv(permutedims(S); rtol=1e-12) * permutedims(M)
        Fl = eigen(Wl)
        VL = Matrix{ComplexF64}(undef, k, k)
        used = falses(k)
        for j in 1:k
            best, bestd = 0, Inf
            for m in 1:k
                used[m] && continue
                d = abs(Fl.values[m] - theta[j])
                if isfinite(d) && d < bestd; bestd, best = d, m; end
            end
            best == 0 && (best = findfirst(!, used))
            used[best] = true
            VL[:, j] = Fl.vectors[:, best]
        end

        # de-mix: rotate the applied block by the Ritz coefficients
        Rnew = MPS[lincomb_mps(VR[:, j], AR;  cutoff=cutoff, maxdim=md) for j in 1:k]
        Lnew = MPS[lincomb_mps(VL[:, j], ATL; cutoff=cutoff, maxdim=md) for j in 1:k]
        for j in 1:k
            nr = norm(Rnew[j]); Rnew[j] = (isfinite(nr) && nr > 1e-300) ? normalize(Rnew[j]) : rand_mps()
            nl = norm(Lnew[j]); Lnew[j] = (isfinite(nl) && nl > 1e-300) ? normalize(Lnew[j]) : rand_mps()
        end
        R, L = Rnew, Lnew

        ntr = min(n_track, k)
        if it > 1 && all(isfinite, theta_prev[1:ntr])
            dtheta = maximum(abs.(theta[1:ntr] .- theta_prev[1:ntr]))
            push!(dtheta_hist, dtheta)
            if dtheta < eps_conv
                reason = "converged"
                break
            end
        end
        theta_prev = copy(theta)

        # refresh a collapsed direction when the pencil is ill-conditioned
        if condS_last > cond_thresh && k >= 2
            jb = k
            r = rand_mps(); l = rand_mps()
            for a in 1:(k-1)
                r = lincomb_mps([1.0, -overlap_noconj(L[a], r)], MPS[r, R[a]]; cutoff=cutoff, maxdim=md)
                l = lincomb_mps([1.0, -overlap_noconj(R[a], l)], MPS[l, L[a]]; cutoff=cutoff, maxdim=md)
            end
            R[jb] = normalize(r); L[jb] = normalize(l)
            reason = (reason == "converged") ? reason : "refreshed"
        end
    end

    theta_eigen = copy(theta)
    for j in 1:k        # bi-orthonormalize so ⟨Lⱼ|Rⱼ⟩ = 1
        ov = overlap_noconj(L[j], R[j])
        if abs(ov) > 1e-10
            L[j] = (1 / sqrt(ov)) * L[j]
            R[j] = (1 / sqrt(ov)) * R[j]
        end
    end

    info = Dict(:niters => niters, :reason => reason,
                :condS => condS_last, :condS_hist => condS_hist,
                :dtheta => dtheta_hist, :theta => theta,
                :theta_eigen => theta_eigen)
    return theta, L, R, info
end

# single-vector LR power method + leading-eigenvalue diagnostic
"""
    run_pm_diagnosed(target_T; p, lambda, dt, maxdim, cutoff, eps_converged,
                     nbeta, MPO_alg, alg, itermax, stuck_after) → NamedTuple

Single-vector `powermethod_lr` wrapper. Returns the bi-normalized (L,R), the
leading Rayleigh-quotient eigenvalue λ₀, the tMPO (reuse it for block_transfer_eigs),
and convergence diagnostics. Kept to illustrate WHY the single-vector method stalls
near the degeneracy (where the block method is needed instead).
"""
function run_pm_diagnosed(target_T::Float64;
        p::Float64=0.1, lambda::Float64=1.0, dt::Float64=0.1,
        maxdim::Int=256, cutoff::Float64=1e-14, eps_converged::Float64=1e-6,
        nbeta::Int=0, MPO_alg::String="VD2", alg::String="RTM",
        itermax::Int=5000, stuck_after::Int=200)

    mpo, scaffold = build_alcaraz_tmpo(target_T; p=p, lambda=lambda, dt=dt,
                                        nbeta=nbeta, MPO_alg=MPO_alg)
    seed_mps = deepcopy(scaffold)                      # random seed -> avoid Z2 trapping
    for i in eachindex(seed_mps)
        seed_mps[i] = randomITensor(ComplexF64, inds(seed_mps[i]))
    end
    normalize!(seed_mps)

    pm_params = PMParams(;
        truncp        = (; cutoff=cutoff, maxdim=maxdim, alg=alg),
        opt_method    = :nosym,
        cutoffs       = [cutoff],
        maxdims       = 2:2:maxdim,
        itermax       = itermax,
        eps_converged = eps_converged,
        normalization = "overlap",
        stuck_after   = stuck_after,
        compute_fidelity = false)

    psi_L, psi_R, pm_info = ITransverse.powermethod_lr(seed_mps, mpo, mpo, pm_params)

    ds_hist  = pm_info[:ds]
    chi_hist = pm_info[:chi]
    niters   = length(ds_hist)
    final_ds = isempty(ds_hist) ? NaN : last(ds_hist)
    stuck    = isempty(ds_hist) || final_ds > eps_converged
    reason   = (!stuck) ? "converged" : (niters >= itermax) ? "maxiter" : "stuck"

    lr_overlap_raw = overlap_lr(psi_L, psi_R)
    lambda0 = expval_LR(psi_L, mpo, psi_R) / lr_overlap_raw

    c = sqrt(lr_overlap_raw)                            # bi-normalize via SCALAR mult (not ./=)
    psi_L = (1/c) * psi_L
    psi_R = (1/c) * psi_R

    return (L=psi_L, R=psi_R, mpo=mpo, scaffold=scaffold, lambda0=lambda0,
            niters=niters, stuck=stuck, reason=reason, final_ds=final_ds,
            ds_hist=ds_hist, chi_hist=chi_hist)
end

# generalized temporal entropies from a converged power method:
"""
    compute_entropies(mp::ModelParams, target_T; scheme, dt, cutoff, maxdim, alg,
                      eps_converged, nbeta, use_block_pm, k_block)
        → NamedTuple(bonds, re, im, L, R, mpo)

Builds the forward tMPO for model `mp` with exponentiation `scheme` (e.g. AlcarazVD2()) and
returns the Rényi-2 temporal-entropy profile S₂(t) = -log Tr(T_t²) per internal bond (Re and Im
separately). Initial state |X+⟩ (free BC). Replaces the old per-model compute_*_entropies drivers.

The boundary tMPS (L,R) come from one of two power methods:
  • `use_block_pm=false` (DEFAULT): single-vector `powermethod_lr` from a random seed. Cheap, but
    STOPS CONVERGING once the transfer-matrix gap closes (the entanglement barrier), so the
    profile is only trustworthy for short/intermediate T (≲5 for Alcaraz p=0.1).
  • `use_block_pm=true`: the oblique block (subspace) method `block_transfer_eigs` with `k_block`
    Ritz vectors, taking the leading (already bi-orthonormal) pair. Stays well-behaved through the
    degeneracy and recovers the conformal dome deeper into the barrier — at higher cost.
"""
function compute_entropies(mp::ModelParams, target_T::Float64;
        scheme::ExpHRecipe, dt::Float64=0.1, cutoff::Float64=1e-12, maxdim::Int=256,
        alg::String="RTM", eps_converged::Float64=1e-6, nbeta::Int=4,
        use_block_pm::Bool=false, k_block::Int=2,
        maxdims::Union{Nothing,AbstractVector{<:Integer}}=nothing)

    Ntime_steps = round(Int, target_T / dt)
    Nsteps      = Ntime_steps + nbeta
    s           = mp.phys_site
    init_state  = complex(state(s, "X+"))

    tp = tMPOParams(mp=mp, dt=dt, nbeta=nbeta, scheme=scheme, dbeta=-im*dt, bl=init_state)
    b  = FwtMPOBlocks(tp)
    spatial_bond_dim = dim(inds(b.Wc, "Site,time")[1])
    time_sites = addtags(siteinds(spatial_bond_dim, Nsteps; conserve_qns=false), "time")

    mpo       = fw_tMPO(b, time_sites, tr=init_state)
    start_mps = fw_tMPS(b, time_sites; tr=init_state, LR=:right)

    if use_block_pm
        # block (subspace) method: robust through the gap closing; returns bi-orthonormal L,R
        _, L_vecs, R_vecs, info = block_transfer_eigs(mpo, start_mps;
            k=k_block, maxdim=maxdim, cutoff=cutoff, itermax=8000, eps_conv=eps_converged,
            maxdims=maxdims)
        info[:reason] in ("maxiter", "stuck") &&
            @warn "block PM did not strictly converge at T=$target_T (reason=$(info[:reason]))"
        psi_L, psi_R = L_vecs[1], R_vecs[1]
    else
        for i in eachindex(start_mps)                  # random seed (anti Z2-trap)
            start_mps[i] = randomITensor(ComplexF64, inds(start_mps[i]))
        end
        normalize!(start_mps)

        pm_params = PMParams(;
            truncp = (; cutoff=cutoff, maxdim=maxdim, alg=alg),
            opt_method = :nosym,
            cutoffs = [cutoff],
            maxdims = 2:2:maxdim,
            itermax = 8000,
            eps_converged = eps_converged,
            normalization = "overlap",
            stuck_after = 2000,
            compute_fidelity = true)

        psi_L, psi_R, _ = ITransverse.powermethod_lr(start_mps, mpo, mpo, pm_params)

        nrm   = overlap_noconj(psi_L, psi_R)           # SCALAR mult, not ./= (avoids norm^N bug)
        psi_L = (1/sqrt(nrm)) * psi_L
        psi_R = (1/sqrt(nrm)) * psi_R
    end

    s2 = ITransverse.gen_renyi2(psi_L, psi_R)
    return (; bonds = 1:length(s2), re = real.(s2), im = imag.(s2),
            L = psi_L, R = psi_R, mpo = mpo)
end

"""
    plot_entropy_profiles(mp, target_times; scheme, dt, ...) → Plots.Plot

Overlays Re(S₂) and Im(S₂) temporal-entropy profiles for a list of target times. Pass
`use_block_pm=true` to use the robust block power method (see `compute_entropies`).
"""
function plot_entropy_profiles(mp::ModelParams, target_times::Vector{Float64};
        scheme::ExpHRecipe, dt::Float64=0.1, cutoff::Float64=1e-12, maxdim::Int=256,
        alg::String="RTM", eps_converged::Float64=1e-6, nbeta::Int=4,
        use_block_pm::Bool=false, k_block::Int=2)

    plt_real = plot(title="Re(S₂)", xlabel="temporal cut t/T", ylabel="Re(S₂)",
                    legend=:outerright, grid=true, framestyle=:box)
    plt_imag = plot(title="Im(S₂)", xlabel="temporal cut t/T", ylabel="Im(S₂)",
                    legend=:outerright, grid=true, framestyle=:box)
    n = length(target_times)
    cr = cgrad(:viridis, n, categorical=true)
    ci = cgrad(:plasma,  n, categorical=true)

    @showprogress "entropy profiles ($(typeof(scheme)))..." for (i, T) in enumerate(target_times)
        res = compute_entropies(mp, T; scheme=scheme, dt=dt, cutoff=cutoff, maxdim=maxdim,
                                alg=alg, eps_converged=eps_converged, nbeta=nbeta,
                                use_block_pm=use_block_pm, k_block=k_block)
        x = range(0.0, 1.0, length=length(res.re))
        lab = "T = $(round(T, digits=1))"
        plot!(plt_real, x, res.re, label=lab, lw=2, color=cr[i])
        plot!(plt_imag, x, res.im, label=lab, lw=2, color=ci[i])
    end
    return plot(plt_real, plt_imag, layout=(1, 2), size=(1200, 450), margin=5Plots.mm)
end

# TDVP Schrödinger Loschmidt amplitude L(T)=⟨ψ0|U(T)|ψ0⟩ (crash-safe)
"""
    tdvp_loschmidt_amplitude(N, target_times; p, lambda, dt, cutoff, maxdim, cachefile)
      → Dict{Float64, NamedTuple}

Evolves |X+⟩^N with TDVP on the Alcaraz Hamiltonian and records the complex Loschmidt
amplitude at each target time. Per-T crash-safe cache (default results/data/...).
Each entry: (G, absG, rate=-log|G|/N, maxchi). target_times sorted, spaced by dt multiples.
"""
function tdvp_loschmidt_amplitude(N::Int, target_times::Vector{Float64};
        p::Float64=0.1, lambda::Float64=1.0, dt::Float64=0.05,
        cutoff::Float64=1e-12, maxdim::Int=256,
        cachefile::Union{String,Nothing}=nothing)

    cf   = isnothing(cachefile) ? "results/data/tdvp_loschmidt_p$(p)_N$(N).jld2" : cachefile
    done = isfile(cf) ? load(cf, "done") : Dict{Float64,Any}()

    sites = siteinds("S=1/2", N)
    psi0  = complex(MPS(sites, "X+"))
    os    = alcaraz_opsum(N, lambda, p)
    H     = MPO(os, sites)

    sorted_Ts  = sort(target_times)
    missing_Ts = [T for T in sorted_Ts if !haskey(done, T)]
    if isempty(missing_Ts)
        @info "All target T values already cached."
        return done
    end

    psi_t = deepcopy(psi0); current_t = 0.0
    for T in sorted_Ts
        steps = round(Int, (T - current_t) / dt)
        steps < 0 && error("target_times must be ascending; got T=$T after t=$current_t")
        for _ in 1:steps
            psi_t = tdvp(H, -im * dt, psi_t; cutoff=cutoff, maxdim=maxdim, nsite=2)
            normalize!(psi_t)
        end
        current_t = T
        haskey(done, T) && (@info "T=$T (cached, evolved through)"; continue)
        G    = inner(psi0, psi_t); absG = abs(G)
        done[T] = (G=G, absG=absG, rate=-log(max(absG, 1e-50))/N, maxchi=maxlinkdim(psi_t))
        jldsave(cf; done)
        @info "T=$T (NEW)  |G|=$(round(absG,digits=5))  χ=$(maxlinkdim(psi_t))"
        GC.gc()
    end
    return done
end

# generic crash-safe sweep: f(T) per T, checkpoint after each
"""
    crashsafe_sweep(f, Ts; cachefile) → done::Dict

Calls `done[T] = f(T)` for each T (sorted), saving a JLD2 checkpoint after every T
and skipping already-cached T. Use a results/data/ path for `cachefile`.
"""
function crashsafe_sweep(f::Function, Ts; cachefile::String)
    done = isfile(cachefile) ? load(cachefile, "done") : Dict{Float64, Any}()
    for T in sort(collect(Ts))
        haskey(done, T) && continue
        try
            done[T] = f(T)
        catch err
            @warn "T=$T failed: $err"
            done[T] = (error=string(err),)
        end
        jldsave(cachefile; done)
        GC.gc()
    end
    return done
end

# save a multi-panel figure to results/imgs/
function plot_panels(panels...; filename::String, title::String="",
                     fig_size::Tuple{Int,Int}=(500*length(panels), 480))
    mkpath("results/imgs")
    plt = plot(panels...; layout=(1, length(panels)), size=fig_size,
               plot_title=title, margin=5Plots.mm)
    savefig(plt, joinpath("results", "imgs", filename))
    return plt
end
