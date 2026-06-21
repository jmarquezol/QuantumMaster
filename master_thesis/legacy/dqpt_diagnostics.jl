# dqpt_diagnostics.jl
# Reusable helpers for the DQPT-diagnostic suite (notebooks 11–16).
# Include via:  includet("dqpt_diagnostics.jl")
# Requires main.jl to have been included first.

using LinearAlgebra, JLD2, Plots

# ─────────────────────────────────────────────────────────────────────────────
# 1. overlap_lr — named alias for the non-conjugating bilinear overlap
# ─────────────────────────────────────────────────────────────────────────────
overlap_lr(L::MPS, R::MPS) = overlap_noconj(L, R)

# ─────────────────────────────────────────────────────────────────────────────
# 2. build_alcaraz_tmpo
#    Returns (mpo, scaffold) without running any power method.
# ─────────────────────────────────────────────────────────────────────────────
function build_alcaraz_tmpo(target_T::Float64;
        p::Float64=0.1, lambda::Float64=1.0, dt::Float64=0.1,
        nbeta::Int=0, MPO_alg::String="VD2")
    Ntime_steps = round(Int, target_T / dt)
    Nsteps      = Ntime_steps + nbeta
    s           = Index(2, "S=1/2")
    init_state  = complex(state(s, "X+"))
    RECIPES     = Dict("WI"=>AlcarazWI(), "WII"=>AlcarazWII(), "VD2"=>AlcarazVD2())
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

# ─────────────────────────────────────────────────────────────────────────────
# 3. lincomb_mps and block_transfer_eigs
#   Truncated linear combination of MPS:  sum_i coeffs[i] * vecs[i]
#   Uses the exact "directsum" combine then SVD-truncates
# ─────────────────────────────────────────────────────────────────────────────
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
                        n_track, cond_thresh) → (theta, L, R, info)

Block (subspace) power method for the leading `k` eigenvalues of the non-Hermitian
transfer operator `mpo`, with separate left/right bases and a fully non-conjugating
(overlap_noconj) oblique Rayleigh-Ritz. Robust through degeneracy.

info keys: :niters, :reason, :condS (final), :condS_hist (per-iteration),
           :dtheta (convergence history), :theta (Ritz values), :theta_eigen
           (same as :theta, labeled for comparison with the Rayleigh quotient).
"""
function block_transfer_eigs(mpo::MPO, scaffold::MPS;
        k::Int=4, maxdim::Int=256, cutoff::Float64=1e-12,
        itermax::Int=300, eps_conv::Float64=1e-8, n_track::Int=2,
        cond_thresh::Float64=1e10)

    sit  = siteinds(scaffold)
    mpoT = swapprime(mpo, 0, 1)   # pure transpose (with NO conjugation / dag) = bilinear-form adjoint : <L| mpo^T |R> = <mpo L| R>

    # Initialize k random MPS vectors for the subspace
    rand_mps() = normalize(complex.(randomMPS(sit, linkdims=2k)))
    R = MPS[rand_mps() for _ in 1:k]  # |R_j>
    L = MPS[rand_mps() for _ in 1:k]  # <L_i|

    # Convergence tracking
    theta       = fill(NaN + 0im, k)   # Current eigenvalues (Ritz values)
    theta_prev  = fill(NaN + 0im, k)   # Previous eigenvalues (for Δθ convergence check)
    dtheta_hist = Float64[]            # History of Δθ per iteration (convergence diagnostic)
    condS_hist  = Float64[]            # History of condition number cond(S) (ill-conditioning diagnostic)
    condS_last  = NaN
    reason      = "maxiter"            # Convergence reason: "converged", "maxiter", "stuck", "refreshed"
    niters      = 0                    # Actual number of iterations run

    for it in 1:itermax
        niters = it

        # STEP 1: Apply the transfer matrix to the current basis (apply one step)
        # AR[j] = MPO · R[j]
        # ATL[j] = MPO^T · L[j] (non-conjugating)
        AR  = MPS[applyn(mpo,  R[j]; cutoff=cutoff, maxdim=maxdim) for j in 1:k]
        ATL = MPS[applyn(mpoT, L[j]; cutoff=cutoff, maxdim=maxdim) for j in 1:k]

        # STEP 2: Build the oblique Rayleigh-Ritz pencil (S, M)
        # The GENERALIZED eigenvalue problem is: M v = λ S v
        #   (equivalently: S^{-1} M v = λ v for standard eigenvalues)
        # This is called an "oblique pencil" because S is non-symmetric (it's a bilinear form)
        S = Matrix{ComplexF64}(undef, k, k)
        M = Matrix{ComplexF64}(undef, k, k)
        for i in 1:k, j in 1:k
            S[i, j] = overlap_noconj(L[i], R[j])      # metric: <L_i|R_j>   (encodes inner product in subspace)
            M[i, j] = overlap_noconj(L[i], AR[j])     # action: <L_i|A|R_j> (encodes dynamics in subspace)
        end

        # Monitor the conditioning of S: if cond(S) -> infinity, the pencil becomes ill-posed
        # High cond(S) happens near-degeneracy (two eigenvalues almost equal in magnitude)
        condS_last = cond(S)
        push!(condS_hist, condS_last)

        # STEP 3: Solve the generalized eigenvalue problem S^{-1} M v = λ v
        # pinv (pseudo-inverse) of S used (instead of inv) to handle near-singular S robustly
        W  = pinv(S; rtol=1e-12) * M
        # Standard eigendecomposition: W vR = λ vR
        Fr = eigen(W)
        # Sort eigenvalues by magnitude (largest first) -> highest λ dominates the long-time dynamics
        permr = sortperm(abs.(Fr.values); rev=true)
        theta = Fr.values[permr]     # Ritz values (eigenvalues in descending |λ| order)
        VR    = Fr.vectors[:, permr] # Right Ritz vectors (coefficients for the new R basis)

        # STEP 4: Compute LEFT Ritz vectors via the transposed pencil
        # Left eigenvalue problem: (M^T)^{-1} (S^T) vL = λ vL  (dual to the right problem)
        # We match left Ritz vectors to right Ritz values by CLOSEST EIGENVALUE,
        # not by eigenvector order (they may come out in different orders).
        Wl = pinv(permutedims(S); rtol=1e-12) * permutedims(M)
        Fl = eigen(Wl)
        VL = Matrix{ComplexF64}(undef, k, k)
        used = falses(k)
        # For each right Ritz value θ[j], find the closest left eigenvalue and use its eigenvector
        for j in 1:k
            best, bestd = 0, Inf
            for m in 1:k
                used[m] && continue
                d = abs(Fl.values[m] - theta[j])  # distance in complex plane
                if isfinite(d) && d < bestd; bestd, best = d, m; end
            end
            # Fallback: if no match found, take the first unused eigenvector
            best == 0 && (best = findfirst(!, used))
            used[best] = true
            VL[:, j] = Fl.vectors[:, best]  # Matched left Ritz vector
        end

        # STEP 5: Update the basis by rotating via Ritz vectors
        # The Ritz vectors (columns of VR and VL) are coefficients in the CURRENT basis:
        #   R_new[j] = ∑_i VR[i,j] * AR[i]  (linear combination of applied vectors)
        #   L_new[j] = ∑_i VL[i,j] * ATL[i]
        # This "de-mixes" the k vectors to align them with the k dominant eigenspaces.
        Rnew = MPS[lincomb_mps(VR[:, j], AR;  cutoff=cutoff, maxdim=maxdim) for j in 1:k]
        Lnew = MPS[lincomb_mps(VL[:, j], ATL; cutoff=cutoff, maxdim=maxdim) for j in 1:k]

        # STEP 6: Normalize the new basis vectors
        # If norm is finite and non-negligible, normalize to unit norm
        # If norm is NaN/Inf/tiny (< 1e-300), the vector collapsed -> reseed with fresh random to recover
        for j in 1:k
            nr = norm(Rnew[j]); Rnew[j] = (isfinite(nr) && nr > 1e-300) ? normalize(Rnew[j]) : rand_mps()
            nl = norm(Lnew[j]); Lnew[j] = (isfinite(nl) && nl > 1e-300) ? normalize(Lnew[j]) : rand_mps()
        end
        R, L = Rnew, Lnew

        # STEP 7: Check convergence
        # We track the leading n_track eigenvalues (typically n_track=2, the top 2)
        # Convergence criterion: Δθ = max|θ_j^{(it)} - θ_j^{(it-1)}| < eps_conv
        # This measures how much the Ritz values changed -> when they stabilize, we've converged
        ntr = min(n_track, k)
        if it > 1 && all(isfinite, theta_prev[1:ntr])
            dtheta = maximum(abs.(theta[1:ntr] .- theta_prev[1:ntr]))
            push!(dtheta_hist, dtheta)
            if dtheta < eps_conv
                reason = "converged"
                break  # Stop iterating -> Ritz values are stable
            end
        end
        theta_prev = copy(theta)

        # STEP 8: Refresh basis if ill-conditioned (near-degeneracy recovery)
        # If cond(S) exceeds cond_thresh (default 1e10), the pencil is nearly singular
        # -> two or more eigenvalues are nearly equal in magnitude (Z2 degeneracy)
        # Solution: Gram-Schmidt the last vector (jb=k) against the first k-1 via overlap_noconj
        # -> the k-th vector stays orthogonal (in the bilinear sense) to the others,
        # preventing numerical collapse into the same eigenspace.
        if condS_last > cond_thresh && k >= 2
            jb = k
            r = rand_mps(); l = rand_mps()
            # Gram-Schmidt: orthogonalize r against R[1..(k-1)] and l against L[1..(k-1)]
            for a in 1:(k-1)
                # r_new <- r - <L_a|r> R_a  (subtract projection onto a-th mode)
                r = lincomb_mps([1.0, -overlap_noconj(L[a], r)], MPS[r, R[a]]; cutoff=cutoff, maxdim=maxdim)
                # l_new <- l - <R_a|l> L_a  (subtract projection onto a-th mode)
                l = lincomb_mps([1.0, -overlap_noconj(R[a], l)], MPS[l, L[a]]; cutoff=cutoff, maxdim=maxdim)
            end
            R[jb] = normalize(r); L[jb] = normalize(l)
            reason = (reason == "converged") ? reason : "refreshed"
        end
    end

    theta_eigen = copy(theta)  
    for j in 1:k
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

# ─────────────────────────────────────────────────────────────────────────────
# 4. run_pm_diagnosed
#    Runs the powermethod_lr cell and returns raw ⟨L|R⟩ overlap BEFORE normalization
#       If ⟨L|R⟩ -> 0 at T*, the left/right fixed points become orthogonal
#       -> Loschmidt zero condition in the transverse picture (DQPT signal)
# ─────────────────────────────────────────────────────────────────────────────
"""
    run_pm_diagnosed(target_T; p, lambda, dt, maxdim, cutoff, eps_converged,
                     nbeta, MPO_alg, alg) → NamedTuple

Fields:
  L, R            MPS          bi-normalized left/right tMPS
  mpo             MPO          tMPO (pass to block_transfer_eigs to avoid rebuild)
  lambda0         ComplexF64   Rayleigh quotient expval_LR(L,mpo,R) / ⟨L|R⟩_raw
  lr_overlap_raw  ComplexF64   ⟨L|R⟩ of the returned states (≈1: overlap-normalization, F1)
  lr_overlap      ComplexF64   ⟨L|R⟩ AFTER explicit bi-normalization (≈ 1)
  lr_cos          Float64      |⟨L̂|R̂⟩| with L̂,R̂ each unit-normalized — Stage D DQPT signal,
                               →0 when the fixed points split across Z2 sectors
  niters          Int
  stuck           Bool         true if last ΔS > eps_converged
  reason          String       "converged" | "maxiter" | "stuck"  (Stage A triage)
  final_ds        Float64      last ΔS
  itermax,stuck_after,eps_converged   echo of the PM settings actually used
  ds_hist         Vector{Float64}
  chi_hist        Vector{Int}
"""
function run_pm_diagnosed(target_T::Float64;
        p::Float64=0.1, lambda::Float64=1.0, dt::Float64=0.1,
        maxdim::Int=256, cutoff::Float64=1e-14, eps_converged::Float64=1e-6,
        nbeta::Int=0, MPO_alg::String="VD2", alg::String="RTM",
        itermax::Int=5000, stuck_after::Int=200)

    mpo, scaffold = build_alcaraz_tmpo(target_T; p=p, lambda=lambda, dt=dt,
                                        nbeta=nbeta, MPO_alg=MPO_alg)

    # random seed to avoid Z2 sector trapping
    seed_mps = deepcopy(scaffold)
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
        compute_fidelity = false
    )

    psi_L, psi_R, pm_info = ITransverse.powermethod_lr(seed_mps, mpo, mpo, pm_params)

    ds_hist  = pm_info[:ds]
    chi_hist = pm_info[:chi]
    niters   = length(ds_hist)
    final_ds = isempty(ds_hist) ? NaN : last(ds_hist)
    stuck    = isempty(ds_hist) || final_ds > eps_converged
    # Reason = converged    if ΔS<eps,
    #          maxiter      if the loop ran the full itermax, 
    #          stuck        otherwise
    reason   = (!stuck)             ? "converged" :
               (niters >= itermax)  ? "maxiter"   : "stuck"

    # overlap
    lr_overlap_raw = overlap_lr(psi_L, psi_R)   # ≈ 1 (always, due to PM normalization)

    # Rayleigh quotient for the leading transfer-matrix eigenvalue
    lambda0 = expval_LR(psi_L, mpo, psi_R) / lr_overlap_raw

    # STAGE D: TRUE DQPT signal through the biorthogonality cosine
    # Key idea: normalize L and R INDEPENDENTLY (breaking the joint normalization)
    # Then measure how "parallel" or "orthogonal" they are via their bilinear overlap
    #
    # Geometric interpretation:
    #   |cos| = |⟨L̂|R̂⟩| / (||L̂|| ||R̂||) ∈ [0, 1]
    #   |cos| ≈ 1  ⟺  L̂ and R̂ are "parallel" (same Z2 sector)
    #   |cos| → 0  ⟺  L̂ and R̂ are "orthogonal" (different Z2 sectors)
    #
    # At the DQPT (Fisher zero), the gap closes and L,R split across Z2 sectors
    # So |⟨L̂|R̂⟩| → 0 is the physical DQPT signal
    #
    # Why norm() and not the generalized norm?
    # The generalized norm = sqrt(⟨L|L⟩) can be complex or zero.
    # We use Euclidean norm (ordinary l2-norm) which is always real, positive, finite.
    Lhat = normalize(psi_L)  # L̂ = L / ||L||_2
    Rhat = normalize(psi_R)  # R̂ = R / ||R||_2
    # |⟨L̂|R̂⟩| / (||L̂|| ||R̂||) — NOTE: since L̂,R̂ are already normalized, the denominator is 1,
    # so this is just |⟨L̂|R̂⟩|. But we write it explicitly to emphasize the biorthogonality structure.
    lr_cos = abs(overlap_lr(Lhat, Rhat)) / (norm(Lhat) * norm(Rhat))

    # Final normalization: bi-normalize for return
    # We scale both L and R by (1/c) where c = sqrt(⟨L_raw|R_raw⟩).
    # This absorbs c into the FIRST tensor only, scaling the MPS norm by 1/c (not c^N).
    # After this step, lr_overlap ≈ 1 (like lr_overlap_raw, but now it's a choice, not forced).
    c = sqrt(lr_overlap_raw)
    psi_L = (1/c) * psi_L  # Scalar multiply (safe): norm(psi_L) ← norm(psi_L) / c
    psi_R = (1/c) * psi_R  # (NOT mps ./= c, which would give norm^N / c)
    lr_overlap = overlap_lr(psi_L, psi_R)   # ≈ 1 after this explicit normalization

    # NB: scaffold is returned so the block method can reuse the SAME tMPO site
    # indices (block_transfer_eigs builds its vectors on siteinds(scaffold), which
    # must match mpo — a freshly rebuilt tMPO has different Index objects).
    return (L=psi_L, R=psi_R, mpo=mpo, scaffold=scaffold,
            lambda0=lambda0,
            lr_overlap_raw=lr_overlap_raw, lr_overlap=lr_overlap,
            lr_cos=lr_cos,
            niters=niters, stuck=stuck, reason=reason, final_ds=final_ds,
            itermax=itermax, stuck_after=stuck_after, eps_converged=eps_converged,
            ds_hist=ds_hist, chi_hist=chi_hist)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. z2_operator and project_parity
#    z2_operator builds the global Z2 flip P = ∏ σˣ as a product MPO (dim-1 links).
#    project_parity = (1 ± P)/2 onto the ±1 sector via directsum-safe MPS arithmetic.
#
#    IMPORTANT: P = ∏ σˣ is only well-defined on spin-1/2 (dim-2) sites. The
#    temporal MPS sites have dimension 1+χ+χ² (VD2), so the Z2 representation in
#    the rotated temporal basis is NOT a simple σˣ string — that is worked out in
#    nb15. z2_operator therefore ERRORS LOUDLY on non-dim-2 sites instead of
#    silently returning something wrong.
# ─────────────────────────────────────────────────────────────────────────────
function z2_operator(sites::Vector{<:Index})
    all(dim(s) == 2 for s in sites) || error(
        "z2_operator: P=∏σˣ needs spin-1/2 (dim-2) sites. These sites have dims " *
        "$(unique(dim.(sites))); the temporal-basis Z2 representation is deferred to nb15.")
    N = length(sites)
    P = MPO(N)
    links = [Index(1, "Link,z2,l=$i") for i in 0:N]
    for i in 1:N
        sx = ITensor(ComplexF64, sites[i]', dag(sites[i]))
        sx[sites[i]' => 1, dag(sites[i]) => 2] = 1.0   # σˣ off-diagonal
        sx[sites[i]' => 2, dag(sites[i]) => 1] = 1.0
        P[i] = sx * onehot(links[i] => 1) * onehot(dag(links[i+1]) => 1)
    end
    return P
end

function project_parity(mps::MPS, sites::Vector{<:Index}; sector::Int=+1)
    sector in (+1, -1) || error("project_parity: sector must be ±1, got $sector")
    P    = z2_operator(sites)
    Pmps = apply(P, mps; cutoff=1e-14)
    # (1 ± P)/2 |ψ⟩ via directsum-safe combination (no density-matrix +)
    return lincomb_mps(ComplexF64[0.5, sector * 0.5], MPS[mps, Pmps]; cutoff=1e-12)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5b. tdvp_loschmidt_amplitude
#     TDVP Schrödinger path for L(T) = ⟨ψ0|U(T)|ψ0⟩ — the normalization- and
#     convergence-free DQPT arbiter (Stage B). 
# ─────────────────────────────────────────────────────────────────────────────
"""
    tdvp_loschmidt_amplitude(N, target_times; p, lambda, dt, cutoff, maxdim)
      → Dict{Float64, NamedTuple}

Runs TDVP on the Alcaraz Hamiltonian from t=0 to max(target_times) and records
the complex Loschmidt amplitude L(T)=⟨ψ0|U(T)|ψ0⟩ at each target time.

Each entry: (G=ComplexF64, absG=Float64, rate=Float64, maxchi=Int)
  G     = ⟨ψ0|ψ(T)⟩  (complex amplitude — no absolute value)
  absG  = |G|
  rate  = -log(max(absG,1e-50)) / N   (intensive Loschmidt rate)
  maxchi= max bond dim reached by TDVP

target_times must be sorted ascending and spaced by multiples of dt.
Cache file: tdvp_loschmidt_p{p}_N{N}.jld2 (crash-safe per target_T).
"""
function tdvp_loschmidt_amplitude(N::Int, target_times::Vector{Float64};
        p::Float64=0.1, lambda::Float64=1.0, dt::Float64=0.05,
        cutoff::Float64=1e-12, maxdim::Int=256,
        cachefile::Union{String,Nothing}=nothing)

    # Setup cache file
    cf = isnothing(cachefile) ? "tdvp_loschmidt_p$(p)_N$(N).jld2" : cachefile
    # Load existing cache if available; if not, start with empty dict
    done = isfile(cf) ? load(cf, "done") : Dict{Float64,Any}()

    # Initialize Hamiltonian and initial state
    sites   = siteinds("S=1/2", N)
    psi0    = complex(MPS(sites, "X+"))  # |X+>^N: product state, ground state at λ→∞
    os      = alcaraz_opsum(N, lambda, p)
    H       = MPO(os, sites)             # ANNNI-type Hamiltonian

    # Determine which T values need to be computed or loaded from cache
    sorted_Ts = sort(target_times)
    missing_Ts = [T for T in sorted_Ts if !haskey(done, T)]  # T values NOT yet cached
    cached_count = length(sorted_Ts) - length(missing_Ts)    # T values that exist in cache

    if cached_count > 0
        @info "Cache smart: $(cached_count)/$(length(sorted_Ts)) T values cached, $(length(missing_Ts)) to compute"
    end

    # Fast path: if all T values are cached, return immediately (no TDVP needed)
    if isempty(missing_Ts)
        @info "All target T values already cached. Done!"
        return done
    end

    # TDVP evolution loops
    psi_t     = deepcopy(psi0)
    current_t = 0.0

    for T in sorted_Ts
        steps = round(Int, (T - current_t) / dt)
        steps < 0 && error("target_times must be sorted ascending; got T=$T after t=$current_t")

        for _ in 1:steps
            psi_t = tdvp(H, -im * dt, psi_t; cutoff=cutoff, maxdim=maxdim, nsite=2)
            normalize!(psi_t)
        end
        current_t = T

        if haskey(done, T)
            @info "T=$T (cached, evolved through)"
            continue
        end

        G     = inner(psi0, psi_t)
        absG  = abs(G)
        rate  = -log(max(absG, 1e-50)) / N
        mchi  = maxlinkdim(psi_t)
        done[T] = (G=G, absG=absG, rate=rate, maxchi=mchi)
        jldsave(cf; done)
        @info "T=$T (NEW)  |G|=$(round(absG,digits=5))  rate=$(round(rate,digits=5))  χ=$mchi"
        GC.gc()
    end

    return done
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. crashsafe_sweep
#    Generic crash-safe loop: calls f(T) for each T, checkpoints after each.
# ─────────────────────────────────────────────────────────────────────────────
"""
    crashsafe_sweep(f, Ts; cachefile) → done::Dict

Calls `result = f(T)` for each T in Ts (sorted), stores in `done[T]`, and
saves a JLD2 checkpoint after every T. Skips already-cached T values.
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

# ─────────────────────────────────────────────────────────────────────────────
# 7. plot_panels — save a multi-panel figure to imgs/
# ─────────────────────────────────────────────────────────────────────────────
function plot_panels(panels...; filename::String, title::String="",
                     fig_size::Tuple{Int,Int}=(500*length(panels), 480))
    mkpath("imgs")
    plt = plot(panels...; layout=(1, length(panels)), size=fig_size,
               plot_title=title, margin=5Plots.mm)
    savefig(plt, joinpath("imgs", filename))
    return plt
end
