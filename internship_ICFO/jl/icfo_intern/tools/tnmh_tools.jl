# tools/tnmh_tools.jl
# =====================================================================
# Reusable primitives for the TNMH (Tensor-Network Metropolis–Hastings)
# mixing-time study.
#
# CONTEXT FOR NEWCOMERS
# ---------------------
# We study the 2D classical Ising model on an Lx × Ly square lattice.
# The Boltzmann target distribution is  π(x) = e^{-β H(x)} / Z.
# An approximate sampler q(x) is built by writing the partition function
# as a 2D PEPS (projected entangled pair state) tensor network, then
# contracting it row-by-row using a "boundary MPS" that is truncated to
# bond dimension D_bound.  This q is used as the PROPOSAL of an
# Independence Metropolis–Hastings (IMH) chain — a Markov chain that
# proposes entirely new configurations from q, independent of the current
# state, and accepts/rejects via the Metropolis ratio  w(x') / w(x)
# where  w(x) = π(x) / q(x)  is the importance weight.
#
# KEY QUANTITIES
#   C = max_x  π(x)/q(x)   — the Mengersen–Tweedie constant.
#                              Controls worst-case mixing: t_mix ~ C log(1/ε).
#                              Liu (1996) proved  λ₂ = 1 − 1/C  exactly for IMH.
#   ρ = per-layer marginal ratio.  Because q factorises over Lx rows,
#       the worst-case weight multiplies:  C ~ ρ^{Lx}.
#       ρ itself depends on (Ly, D_bound, β).
#   χ²(π‖q) = Var_q(w) = E_q[w²] − 1  — average-case mismatch, governs
#       typical mixing and explains why practice is fast even when C is large.
#
# ITENSORS CRASH COURSE (for readers unfamiliar with the library)
# ---------------------------------------------------------------
# - `Index(dim, tags)`:  a named dimension.  Tags like "Site,x=3" are labels.
# - `ITensor(vals, ind)`:  a tensor with named indices.  Contraction is done
#   by multiplying (`*`): shared indices are summed over automatically.
# - `MPS` (Matrix Product State):  a 1D chain of tensors with "virtual" bond
#   indices connecting neighbours and a "physical" index per site.
# - `MPO` (Matrix Product Operator):  like MPS but with two physical indices
#   per site (input + output).  `apply(MPO, MPS)` contracts them.
# - `replaceinds(T, old => new)`:  relabels indices so tensors can be connected.
# - `noprime(T)`:  removes the prime level from all indices (cleans up after
#   MPO application, which primes the output indices).
# - `orthogonalize!(mps, site)`:  gauge-transforms the MPS so the orthogonality
#   centre sits at `site`, making norms/overlaps easy to extract there.
# - `normalize!(mps)`:  rescales the MPS to have norm 1.
# - `scalar(T)`:  extracts the single number from a 0-index ITensor.
#
# ASSUMES `include("main.jl")` has already been called, which provides:
#   create_ising_peps, compute_bottom_envs, sample_classical_1d,
#   sample_config_opt, measure_energy.
# =====================================================================

using LinearAlgebra, Random, Statistics

# Fix the random seed for reproducibility across all functions here.
set_seed(s::Integer) = Random.seed!(s)


# =====================================================================
# UTILITY: numerically stable log-sum-exp
# =====================================================================
# MOTIVATION: We frequently need  log(Σ exp(vᵢ)).  Naively exponentiating
# large or very negative numbers causes overflow/underflow.  The standard
# trick: factor out the maximum, compute in a shifted domain, then add it back.
#   log Σ exp(vᵢ) = m + log Σ exp(vᵢ − m),   where m = max(v).
# After the shift, the largest exponent is exp(0) = 1, so no overflow.

function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)                   # shift constant = largest element
    isfinite(m) || return m          # guard: if all -Inf, return -Inf
    return m + log(sum(exp.(v .- m)))  # shifted sum is safe from overflow
end


# #####################################################################
#                     PHASE 0 — INFRASTRUCTURE
# #####################################################################
# The repo lacked a way to evaluate  log q(x)  for an ARBITRARY fixed
# configuration x (the sampler draws random configs, but never evaluates
# the probability of a given one).  This is the single most important
# primitive: without it we cannot compute importance weights  w = π/q,
# the Mengersen–Tweedie constant  C = max w, or build the IMH kernel.
# #####################################################################


# =====================================================================
# Deterministic log q(row | environment) for a PINNED row of spins
# =====================================================================
# MOTIVATION: The sampler in main.jl (`sample_classical_1d`) draws spins
# randomly from the conditional distribution of one row given its
# environment.  Here we need the same conditional probability, but for a
# SPECIFIC configuration (no randomness) — we "pin" each spin to its
# given value and accumulate  Σ log p(spin_y | spins_{1..y-1}, env).
#
# HOW IT WORKS (the chain-rule factorisation):
# The effective row MPS encodes unnormalised weights for each spin config.
# We sweep left→right, at each site computing the unnormalised weight for
# spin=0 (w0) and spin=1 (w1).  The conditional probability of the chosen
# spin is  p_chosen = w_chosen / (w0 + w1).  After choosing, we "collapse"
# the physical index to the chosen spin and advance the left environment.
#
# The right environments R[y] are precomputed right→left: R[y] represents
# the contraction of sites y..Ly with their physical indices summed out
# (traced).  This lets us evaluate w0, w1 at each site in O(1) contractions
# instead of recomputing the full right side each time.

function sample_classical_1d_pinned(row_mps::MPS, s_inds, pinned_spins)
    Ly = length(row_mps)     # number of columns = number of sites in the row
    log_prob_row = 0.0       # accumulator for log q(row | env)

    # --- Step 1: Precompute right environments R[y] (right → left) ---
    # R[y] = contraction of row_mps sites y..Ly with physical indices traced out.
    # "Traced out" means we sum over both spin values, done by contracting with
    # the vector [1, 1] on the physical index (this is the "partial trace" in
    # tensor-network language — it sums over the physical degree of freedom).
    R = Vector{ITensor}(undef, Ly)
    temp = ITensor(1.0)                   # start with a scalar 1 (no indices)
    for y in Ly:-1:1
        # Contract site y's physical index with [1,1] to trace it out
        T_traced = row_mps[y] * ITensor([1.0, 1.0], s_inds[y])
        temp *= T_traced                  # contract with the running right env
        n = norm(temp); if n > 0; temp ./= n; end  # normalise for numerical stability
        R[y] = temp
    end

    # --- Step 2: Sweep left → right, pinning each spin ---
    L_env = ITensor(1.0)     # left environment, starts as scalar 1 (nothing to the left)
    for y in 1:Ly
        T = row_mps[y]       # the MPS tensor at site y (has physical + bond indices)

        # Build spin projectors: [1,0] projects onto spin=0, [0,1] onto spin=1.
        # In ITensors, contracting T with a projector on the physical index s_inds[y]
        # "fixes" that spin value and removes the physical index from the result.
        proj0 = ITensor([1.0, 0.0], s_inds[y])
        proj1 = ITensor([0.0, 1.0], s_inds[y])

        # Compute the unnormalised weight for each spin value:
        #   w0 = L_env · (T with spin=0) · R[y+1]
        # This is the total weight of all configs that have spin=0 at site y,
        # given everything to the left (L_env) and right (R[y+1]).
        # At the last site (y=Ly) there is no right neighbour, so R → scalar 1.
        w0 = max(0.0, scalar(L_env * (T * proj0) * (y < Ly ? R[y+1] : ITensor(1.0))))
        w1 = max(0.0, scalar(L_env * (T * proj1) * (y < Ly ? R[y+1] : ITensor(1.0))))

        # Conditional probability of spin=0 at this site
        total_w = w0 + w1
        p0 = total_w < 1e-15 ? 0.5 : w0 / total_w   # guard against zero (shouldn't happen)

        # Look up the probability of the PINNED (given) spin value
        chosen_spin = pinned_spins[y]
        p_chosen = chosen_spin == 0 ? p0 : (1.0 - p0)

        # Accumulate log-probability (chain rule: log q = Σ log p_chosen)
        log_prob_row += p_chosen > 1e-15 ? log(p_chosen) : -Inf

        # "Collapse" the physical index: contract T with the chosen projector
        # to update the left environment for the next site.
        chosen_proj = chosen_spin == 0 ? proj0 : proj1
        L_env *= (T * chosen_proj)
        n = norm(L_env); if n > 0; L_env ./= n; end   # renormalise for stability
    end

    return log_prob_row  # = log q(this row | top_env, bottom_env)
end


# =====================================================================
# Build the Ising PEPS and bottom boundary-MPS environments (once)
# =====================================================================
# MOTIVATION: For enumeration (looping over all 2^N configs), we want to
# build the PEPS tensors and precompute the bottom environments only ONCE,
# then reuse them for every config.  This avoids redundant O(Lx·Ly) work
# per config.

function build_peps_envs(Lx::Int, Ly::Int, beta::Float64, D_bound::Int, J::Float64=1.0)
    # create_ising_peps: builds the Ising PEPS tensors A[x,y], physical
    # indices s_inds[x,y], and vertical bond indices v_inds[x,y].
    A, s_inds, v_inds = create_ising_peps(Lx, Ly, beta, J)

    # compute_bottom_envs: precomputes boundary MPS from the bottom,
    # sweeping upward row by row with truncation to D_bound.
    # bottom_envs[x] represents the contraction of rows x..Lx,
    # traced over their physical indices and compressed.
    bottom_envs = compute_bottom_envs(A, s_inds, v_inds, Lx, Ly, D_bound)

    return A, s_inds, v_inds, bottom_envs
end


# =====================================================================
# Deterministic log q(config) for a full 2D configuration (low-level)
# =====================================================================
# MOTIVATION: This is the key new primitive — the "pinned" twin of
# sample_config_opt (main.jl:147).  Instead of DRAWING spins randomly
# from the sequential conditional distributions, it EVALUATES the
# probability of a GIVEN configuration.  The math is identical:
#   log q(config) = Σ_{x=1}^{Lx} log q(row_x | rows_{1..x-1}, bottom_env)
# using the chain rule.  Each row's contribution comes from
# sample_classical_1d_pinned.
#
# HOW THE 2D CONTRACTION WORKS:
# We sweep rows top → bottom.  For each row x:
#   1. Build the "effective row MPS" by sandwiching row x's PEPS tensors
#      between the running top_env (rows already processed above) and
#      bottom_envs[x+1] (precomputed environment of rows below).
#   2. Evaluate the pinned log-probability for this row's spins.
#   3. Fold the pinned row into top_env: fix the physical indices to the
#      given spins (making the row into an MPO), apply to top_env, and
#      compress to D_bound.  Now top_env represents rows 1..x.
#
# The `!` suffix (Julia convention) signals this function MUTATES nothing
# but takes pre-built data — it's the "in-place" efficiency variant.

function proposal_logprob!(config, A, s_inds, v_inds, bottom_envs, Lx::Int, Ly::Int, D_bound::Int)
    top_env = nothing    # will become the boundary MPS representing rows above

    # We need "bridge" indices `b[y]` to connect top_env (which has already
    # absorbed previous rows) to the current row's vertical indices.
    # These are fresh Index objects that top_env's MPS uses at its open edge.
    b = [Index(2, "Site,b=$y") for y in 1:Ly]

    log_prob_tot = 0.0   # accumulator for the full-lattice log q

    for x in 1:Lx
        # --- Build the effective row MPS for row x ---
        # For each site (x,y), start with the PEPS tensor A[x,y], then
        # contract it with the top environment (if rows above have been
        # processed) and the bottom environment (precomputed below).
        # The result has only the physical index s[x,y] and horizontal
        # bonds — an effective 1D MPS for this row.
        eff_tensors = ITensor[]
        for y in 1:Ly
            T = A[x, y]   # PEPS tensor at site (x,y)

            # Attach top environment: top_env[y] lives on bridge index b[y],
            # but we need it to connect to v_inds[x-1,y] (the vertical bond
            # between row x-1 and row x).  replaceinds relabels accordingly.
            if top_env !== nothing
                T_top = replaceinds(top_env[y], b[y] => v_inds[x-1, y])
                T *= T_top   # contract: sums over the vertical bond to row above
            end

            # Attach bottom environment: bottom_envs[x+1] represents rows
            # (x+1)..Lx already traced/compressed.  Contracting it with T
            # closes the vertical bond to the row below.
            if x < Lx
                T *= bottom_envs[x+1][y]
            end

            push!(eff_tensors, T)
        end
        # Wrap as an MPS — this is the effective 1D problem for row x
        row_mps = MPS(eff_tensors)

        # Evaluate log q(row_x | environment) using the pinned sampler
        log_prob_tot += sample_classical_1d_pinned(row_mps, s_inds[x, :], config[x, :])

        # --- Fold row x into top_env (update the running top environment) ---
        # Fix the physical indices to the given spin values by contracting
        # each PEPS tensor with a projector [1,0] or [0,1].  The result is
        # a row of tensors with no physical index — an MPO-like object that,
        # when applied to top_env, advances the boundary by one row.
        mpo_tensors = ITensor[]
        for y in 1:Ly
            proj = ITensor(config[x, y] == 0 ? [1.0, 0.0] : [0.0, 1.0], s_inds[x, y])
            push!(mpo_tensors, A[x, y] * proj)   # project out the physical index
        end

        if top_env === nothing
            # First row (x=1): there's no existing top_env to apply the MPO to.
            # The pinned row IS the initial top_env.  We relabel its vertical
            # indices to the bridge indices b[y] so subsequent rows can connect.
            top_tensors = ITensor[]
            for y in 1:Ly
                push!(top_tensors, replaceinds(mpo_tensors[y], [v_inds[1, y]] => [b[y]]))
            end
            top_env = MPS(top_tensors)
            normalize!(top_env)   # keep the overall scale O(1) for stability
        elseif x < Lx
            # Interior row: apply the pinned row as an MPO to the existing top_env.
            # The MPO has two vertical indices (one connecting to top_env via
            # v_inds[x-1,y], one going down via v_inds[x,y]).  We relabel them
            # to (b[y], b[y]') so ITensors' `apply` function recognises which
            # index is "input" (matching top_env's b[y]) and which is "output"
            # (becoming the new b[y] after noprime).
            mpo_tensors_replaced = ITensor[]
            for y in 1:Ly
                T = replaceinds(mpo_tensors[y], [v_inds[x-1, y], v_inds[x, y]] => [b[y], b[y]'])
                push!(mpo_tensors_replaced, T)
            end
            # apply(MPO, MPS) contracts the MPO with the MPS, producing a new MPS.
            # maxdim=D_bound truncates the bond dimension (SVD compression).
            # noprime removes the prime level from the output indices.
            top_env = noprime(apply(MPO(mpo_tensors_replaced), top_env; maxdim=D_bound, cutoff=1e-10))
            normalize!(top_env)
        end
        # (At the last row x=Lx we don't need to update top_env — we're done.)
    end

    return log_prob_tot  # = log q(full config) under the D_bound-truncated proposal
end


# =====================================================================
# Convenience wrapper: builds PEPS + envs, then evaluates log q
# =====================================================================
# MOTIVATION: When you just need log q for one config and don't care
# about reuse, this handles setup internally.  For loops over many
# configs (enumeration), use build_peps_envs + proposal_logprob! instead.

function proposal_logprob(config, Lx::Int, Ly::Int, beta::Float64, D_bound::Int, J::Float64=1.0)
    A, s_inds, v_inds, bottom_envs = build_peps_envs(Lx, Ly, beta, D_bound, J)
    return proposal_logprob!(config, A, s_inds, v_inds, bottom_envs, Lx, Ly, D_bound)
end


# =====================================================================
# Per-row CUMULATIVE log-probability  (Miguel's "accumulation" view)
# =====================================================================
# MOTIVATION: proposal_logprob! returns only the TOTAL log q(config).
# But the chain rule says log q = Σ_x log q(row_x | rows above), so the
# *running* sum after pinning the first k rows is itself meaningful — it
# is the log-marginal probability of the top k rows under the proposal.
# Tracking how this accumulates row by row is exactly what lets us watch
# the importance-weight error build up "as the number of pinned spins
# increases" (Miguel's framing).
#
# This is identical to proposal_logprob! except it records cum[x] = the
# running log-prob after row x, and returns the whole length-Lx vector.

function accumulation_logprobs!(config, A, s_inds, v_inds, bottom_envs, Lx::Int, Ly::Int, D_bound::Int)
    top_env = nothing
    b = [Index(2, "Site,b=$y") for y in 1:Ly]
    cum = zeros(Float64, Lx)   # cum[x] = log (proposal marginal of rows 1..x)
    running = 0.0

    for x in 1:Lx
        # --- build the effective row MPS (same as proposal_logprob!) ---
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

        # accumulate this row's conditional log-prob into the running total
        running += sample_classical_1d_pinned(row_mps, s_inds[x, :], config[x, :])
        cum[x] = running           # snapshot the running sum after row x

        # --- fold row x into top_env (same as proposal_logprob!) ---
        mpo_tensors = ITensor[]
        for y in 1:Ly
            proj = ITensor(config[x, y] == 0 ? [1.0, 0.0] : [0.0, 1.0], s_inds[x, y])
            push!(mpo_tensors, A[x, y] * proj)
        end
        if top_env === nothing
            top_tensors = [replaceinds(mpo_tensors[y], [v_inds[1, y]] => [b[y]]) for y in 1:Ly]
            top_env = MPS(top_tensors); normalize!(top_env)
        elseif x < Lx
            mpo_tensors_replaced = [replaceinds(mpo_tensors[y], [v_inds[x-1, y], v_inds[x, y]] => [b[y], b[y]']) for y in 1:Ly]
            top_env = noprime(apply(MPO(mpo_tensors_replaced), top_env; maxdim=D_bound, cutoff=1e-10))
            normalize!(top_env)
        end
    end
    return cum   # cum[x] = log q(rows 1..x)  (running marginal log-prob)
end


# =====================================================================
# Accumulation curve:  log r_k = log[ π(rows 1..k) / q(rows 1..k) ]
# =====================================================================
# MOTIVATION: this is THE object in Miguel's "watch the error accumulate"
# picture.  For a fixed configuration, we track the running log-ratio of
# the exact marginal to the truncated-proposal marginal as we pin more
# rows.  The trick: the proposal with bond dimension D = 2^Ly is *exact*
# (no truncation ⇒ q = π), so running it at D = 2^Ly gives log π(rows 1..k)
# and running it at the working D gives log q(rows 1..k).  Their difference
# is the running log-ratio.
#
#   log r_k = Σ_{i≤k} δ_i,   δ_i = log[π(row_i|above)/q(row_i|above)]
#
# Endpoints:  r_0 = 1 (log r_0 = 0);  r_{Lx} = π(x)/q(x) = w(x), so the
# final value is exactly the log importance weight of this configuration.
# For the worst-case config s★, log r_{Lx} = log C.  A typical config stays
# flat near 0; s★ climbs (≈ linearly once the environment saturates), and
# the slope of that climb is the per-row rate log ρ.
#
# Returns the length-Lx vector [log r_1, …, log r_{Lx}].

function accumulation_curve(config, Lx::Int, Ly::Int, beta::Float64, D_bound::Int; J::Float64=1.0)
    # truncated proposal → running log q(rows 1..k)
    A, s, v, be = build_peps_envs(Lx, Ly, beta, D_bound, J)
    logq_run = accumulation_logprobs!(config, A, s, v, be, Lx, Ly, D_bound)
    # exact (D = 2^Ly, no truncation) → running log π(rows 1..k)
    Ae, se, ve, bee = build_peps_envs(Lx, Ly, beta, 2^Ly, J)
    logpi_run = accumulation_logprobs!(config, Ae, se, ve, bee, Lx, Ly, 2^Ly)
    return logpi_run .- logq_run   # log r_k, k = 1..Lx
end


# #####################################################################
#                 PHASE 0.3 — FULL ENUMERATION HARNESS
# #####################################################################
# MOTIVATION: To compute the TRUE Mengersen–Tweedie constant C = max_x π(x)/q(x),
# we need to evaluate π and q for EVERY configuration x ∈ {0,1}^{Lx·Ly}.
# This is feasible only for small lattices (2^N configs), but gives us
# exact ground truth: C, the full weight distribution, the partition
# function Z, and the worst-case config.  This is the backbone of
# Phases 1–2, where we verify Liu's identity and the C ~ ρ^Lx law.
# #####################################################################


# =====================================================================
# Integer ↔ spin-configuration bijection
# =====================================================================
# MOTIVATION: We enumerate configs by looping i = 0, 1, ..., 2^N − 1.
# Each integer i encodes a spin config via its binary representation.
# Column-major ordering: bit k maps to site (x, y) with
#   x = (k mod Lx) + 1,  y = (k ÷ Lx) + 1.
# This matches Julia's column-major array layout.

function config_from_int(i::Integer, Lx::Int, Ly::Int)
    config = zeros(Int, Lx, Ly)
    for k in 1:(Lx * Ly)
        x = ((k - 1) % Lx) + 1            # row index (1-based)
        y = ((k - 1) ÷ Lx) + 1            # column index (1-based)
        config[x, y] = (i >> (k - 1)) & 1  # extract bit k from integer i
    end
    return config
end


# =====================================================================
# Enumerate all 2^N configs → exact C, π, q, w arrays
# =====================================================================
# MOTIVATION: This is the brute-force "oracle" — iterate every possible
# spin configuration, compute its energy E(x), exact Boltzmann weight
# log π(x) = −βE(x) − log Z, and proposal log-probability log q(x).
# From these we get the importance weight  w(x) = π(x)/q(x)  and the
# Mengersen–Tweedie constant  C = max_x w(x).
#
# Also returns sum_q = Σ_x q(x) as a sanity check — it must equal 1.0
# if the proposal is a valid probability distribution (Phase 0 control).
#
# RETURNS a NamedTuple with fields:
#   logq, logpi, logw, E, w  — arrays of length 2^N (one entry per config)
#   C       — the Mengersen–Tweedie constant  max_x w(x)
#   sum_q   — normalisation check  Σ_x q(x)  (should be ≈ 1.0)
#   logZ    — log partition function  log Σ_x e^{-βE(x)}
#   argmax  — index (1-based) of the worst-case config  argmax_x w(x)

function enumerate_weights(Lx::Int, Ly::Int, beta::Float64, D_bound::Int; J::Float64=1.0)
    N = Lx * Ly
    # Safety: 2^22 ≈ 4M configs is about the practical limit on a laptop
    N <= 22 || error("enumerate_weights: 2^$N too large; keep Lx·Ly ≲ 18–20")
    M = 1 << N    # total number of configs = 2^N

    # Build the PEPS and bottom environments once (reused for every config)
    A, s_inds, v_inds, bottom_envs = build_peps_envs(Lx, Ly, beta, D_bound, J)

    logq = zeros(Float64, M)   # log q(x) for each config
    E = zeros(Float64, M)      # energy E(x) for each config
    for i in 0:(M - 1)
        cfg = config_from_int(i, Lx, Ly)    # decode integer → Lx×Ly spin matrix
        # Evaluate the deterministic proposal probability for this config
        logq[i+1] = proposal_logprob!(cfg, A, s_inds, v_inds, bottom_envs, Lx, Ly, D_bound)
        # Compute the Ising energy  H = −J Σ_{⟨ij⟩} s_i s_j
        E[i+1] = measure_energy(cfg, J)
    end

    # Compute exact Boltzmann probabilities:
    #   log π(x) = −β E(x) − log Z,  where  Z = Σ_x e^{-β E(x)}
    logpi_un = -beta .* E                  # unnormalised log-Boltzmann weights
    logZ = logsumexp(logpi_un)             # log Z via stable log-sum-exp
    logpi = logpi_un .- logZ               # normalised: log π(x) = log(e^{-βE}/Z)

    # Importance weights:  log w(x) = log π(x) − log q(x)
    logw = logpi .- logq
    w = exp.(logw)

    return (logq=logq, logpi=logpi, logw=logw, E=E, w=w,
            C=maximum(w),             # Mengersen–Tweedie constant
            sum_q=sum(exp.(logq)),     # normalisation check (should be 1.0)
            logZ=logZ,                 # log partition function
            argmax=argmax(w))          # worst-case config (1-based index)
end


# #####################################################################
#             PHASE 1 — IMH TRANSITION KERNEL + SPECTRUM
# #####################################################################
# MOTIVATION: Liu (1996) proved that for an Independence Metropolis–
# Hastings chain, the second-largest eigenvalue of the transition kernel
# is EXACTLY  λ₂ = 1 − 1/C.  This means the spectral gap γ = 1/C and
# the relaxation time t_rel = C — not just bounded by C, but equal to it.
# We verify this by explicitly building the full 2^N × 2^N transition
# matrix and diagonalising it.
# #####################################################################


# =====================================================================
# Build the dense IMH transition kernel P[a→b]
# =====================================================================
# MOTIVATION: The transition probability for an IMH chain proposing
# state b from state a is:
#   P[a,b] = q(b) · min(1, w(b)/w(a))    for b ≠ a
#   P[a,a] = 1 − Σ_{b≠a} P[a,b]         (probability of staying)
#
# The min(1, w(b)/w(a)) is the Metropolis acceptance probability.
# If the proposed state has a higher importance weight, we always accept;
# otherwise we accept with probability w(b)/w(a).
#
# We build this as a dense M×M matrix (M = 2^N).  Feasible up to
# M ≈ 4096 (N=12), which requires ~128 MB.

function imh_kernel(logq::AbstractVector, logw::AbstractVector)
    M = length(logq)
    # Warn if the matrix will be very large (>4096 states = 128 MB)
    M <= 4096 || @warn "imh_kernel: dense $M×$M matrix is large ($(round(M^2*8/1e9,digits=2)) GB)"

    q = exp.(logq)   # proposal probabilities (vector of length M)
    P = zeros(Float64, M, M)

    @inbounds for a in 1:M
        s = 0.0       # accumulator for the off-diagonal row sum
        for b in 1:M
            if a != b
                # MH acceptance:  min(1, w(b)/w(a)) = min(1, exp(logw[b]−logw[a]))
                # Transition prob = proposal × acceptance
                P[a, b] = q[b] * min(1.0, exp(logw[b] - logw[a]))
                s += P[a, b]
            end
        end
        # Diagonal = probability of rejection (staying in state a)
        P[a, a] = 1.0 - s
    end
    return P
end


# =====================================================================
# Extract λ₂ (second-largest eigenvalue) from the kernel
# =====================================================================
# MOTIVATION: The leading eigenvalue of any stochastic matrix is 1
# (corresponding to the stationary distribution π).  The SECOND largest
# eigenvalue λ₂ controls the rate of convergence:
#   ‖Pⁿ(x,·) − π‖_TV  ≤  λ₂ⁿ
# For IMH, Liu proved  λ₂ = 1 − 1/C  exactly.  We verify this.

function lambda2(P::AbstractMatrix)
    ev = sort(real.(eigvals(P)), rev=true)  # all eigenvalues, sorted descending
    return ev[2]                            # second-largest (ev[1] should be ≈ 1.0)
end


# =====================================================================
# Total variation distance between two distributions
# =====================================================================
# MOTIVATION: TV distance  d_TV(p, π) = ½ Σ_x |p(x) − π(x)|  measures
# how far a distribution p is from the target π.  It's the standard
# convergence metric for MCMC: the chain has "mixed" when d_TV < ε.

tv_distance(p::AbstractVector, pivec::AbstractVector) = 0.5 * sum(abs.(p .- pivec))


# =====================================================================
# TV-decay curve: iterate Pⁿ from a worst-case start
# =====================================================================
# MOTIVATION: Starting from the worst-case configuration x★ = argmax w(x)
# (a point mass δ_{x★}), we repeatedly apply the transition kernel P and
# measure how fast the distribution converges to π.  The resulting curve
# ‖Pⁿ(x★,·) − π‖_TV  vs  n  should follow  (1 − 1/C)ⁿ  exactly (Liu),
# until floating-point precision runs out (~1e-13 for Float64).

function tv_decay(P::AbstractMatrix, pivec::AbstractVector, x0::Int, nmax::Int)
    M = size(P, 1)
    p = zeros(M); p[x0] = 1.0     # start with all mass on state x0
    out = zeros(Float64, nmax)
    for n in 1:nmax
        # Left-multiply by P: p_{n+1} = p_n^T · P  (row-vector convention).
        # This advances the distribution by one step of the Markov chain.
        p = vec(p' * P)
        out[n] = tv_distance(p, pivec)
    end
    return out    # out[n] = ‖Pⁿ(x0,·) − π‖_TV
end


# #####################################################################
#           PHASE 2 — ρ(Ly, D, β): PER-LAYER MARGINAL RATIO
# #####################################################################
# MOTIVATION: The per-layer rate ρ is the quantity the old notebook
# (simulation.ipynb) mislabels "C".  It measures how much the TRUNCATED
# boundary MPS distorts the single-row marginal relative to the EXACT
# boundary MPS, at the fixed-point (plateau) after many rows.
#
# ρ matters because  C ~ ρ^{Lx}  — the full-lattice worst-case weight
# is the per-layer ratio raised to the power of the number of rows.
# So  log C ~ Lx · log ρ,  and  log ρ  depends on (Ly, D, β).
#
# The cleanly-named functions below are a non-destructive reimplementation
# of the notebook's compute_asymptotic_C / run_error_tracker.
# simulation.ipynb is left untouched as the historical record.
# #####################################################################


# =====================================================================
# Compute the plateau value of ρ (saturated per-layer ratio)
# =====================================================================
# MOTIVATION: We run TWO boundary MPS side by side — one exact (bond
# dim 2^Ly, lossless) and one truncated (bond dim D_bound) — from the
# bottom of the strip upward for Lx rows until they saturate.  At the
# top row (row 1, which has no upper bond), we brute-force over all
# 2^Ly single-row spin configs and take the max ratio of exact/truncated
# single-row probabilities.  This is ρ (per-layer), NOT C (full-lattice).
#
# WHY TWO ENVIRONMENTS:
# The exact environment captures the true Boltzmann distribution of the
# half-plane below; the truncated one is what the proposal sampler
# actually uses.  Their ratio on any single-row marginal tells us how
# badly truncation distorts the proposal at that row.
#
# WHY THE TOP ROW:
# At row 1 there is no upper vertical bond, so no need for a [1,1] cap
# (the marginal-vs-conditional ambiguity of §9.2 doesn't apply here).
# The measurement is cleanest at the top.

function compute_rho_plateau(Lx::Int, Ly::Int, beta::Float64, D_bound::Int; J::Float64=1.0)
    A, s_inds, v_inds = create_ising_peps(Lx, Ly, beta, J)

    # Bridge indices for the boundary MPS (see proposal_logprob! for explanation)
    b = [Index(2, "Site,b=$y") for y in 1:Ly]

    # --- Initialise both environments from the bottom row (row Lx) ---
    # Trace over the bottom row's physical indices ([1,1] sums both spins)
    # and relabel the upward vertical bonds to bridge indices b[y].
    tensors_Lx = ITensor[]
    for y in 1:Ly
        T = A[Lx, y] * ITensor([1.0, 1.0], s_inds[Lx, y])  # trace physical
        T = replaceinds(T, [v_inds[Lx-1, y]] => [b[y]])     # relabel to bridge
        push!(tensors_Lx, T)
    end
    E_exact = MPS(tensors_Lx)     # exact boundary MPS (will keep full bond dim)
    E_trunc = MPS(tensors_Lx)     # truncated boundary MPS (will compress to D_bound)

    # Log-Z accumulators: track the normalisation constants pulled out by
    # orthogonalize! so we can compare ABSOLUTE (not relative) probabilities.
    logZe = 0.0; logZt = 0.0

    # The exact environment needs bond dimension 2^Ly to be truly lossless.
    exact_maxdim = 2^Ly

    # --- Sweep upward from row Lx-1 to row 2, advancing both environments ---
    for x in Lx-1:-1:2
        # Build the transfer MPO for row x: trace over physical index,
        # relabel vertical bonds to (b[y], b[y]') for the apply() interface.
        tensors_mpo = ITensor[]
        for y in 1:Ly
            T = A[x, y] * ITensor([1.0, 1.0], s_inds[x, y])   # trace physical
            T = replaceinds(T, [v_inds[x, y], v_inds[x-1, y]] => [b[y], b[y]'])
            push!(tensors_mpo, T)
        end
        MPOx = MPO(tensors_mpo)

        # Advance the exact environment (maxdim = 2^Ly: no truncation)
        E_exact = noprime(apply(MPOx, E_exact; maxdim=exact_maxdim, cutoff=1e-15))
        # Pull out the norm at site 1 to keep numbers O(1) and track log Z
        orthogonalize!(E_exact, 1); ne = norm(E_exact[1]); E_exact[1] ./= ne; logZe += log(ne)

        # Advance the truncated environment (maxdim = D_bound: SVD truncation)
        E_trunc = noprime(apply(MPOx, E_trunc; maxdim=D_bound, cutoff=1e-10))
        orthogonalize!(E_trunc, 1); nt = norm(E_trunc[1]); E_trunc[1] ./= nt; logZt += log(nt)
    end

    # --- Measure ρ at the top row (row 1) ---
    # Brute-force: loop over all 2^Ly spin configs for row 1, compute
    # the probability under exact and truncated environments, take the max ratio.
    max_ratio = 0.0
    x = 2   # E_exact/E_trunc sit at row 2's upper edge; we measure row 1 (x-1 = 1)
    for i in 0:(2^Ly - 1)
        spins = [(i >> (j - 1)) & 1 for j in 1:Ly]    # decode integer → spin vector
        rte = ITensor[]; rtt = ITensor[]                 # exact and truncated contractions
        for y in 1:Ly
            T = A[x-1, y]     # PEPS tensor at row 1, site y
            # Pin the physical index to the current spin value
            sv = spins[y] == 0 ? [1.0, 0.0] : [0.0, 1.0]
            T *= ITensor(sv, s_inds[x-1, y])
            # Contract with the boundary environment (relabelled from b to v_inds)
            push!(rte, T * replaceinds(E_exact[y], b[y] => v_inds[x-1, y]))
            push!(rtt, T * replaceinds(E_trunc[y], b[y] => v_inds[x-1, y]))
        end
        # Contract all sites along the row → a scalar (the unnormalised probability)
        pe = abs(scalar(reduce(*, rte)))    # exact probability (unnormalised)
        pt = abs(scalar(reduce(*, rtt)))    # truncated probability (unnormalised)
        if pt > 1e-30 && pe > 1e-30
            # Restore absolute scale using the accumulated log-Z factors
            r = exp((log(pe) + logZe) - (log(pt) + logZt))
            r > max_ratio && (max_ratio = r)
        end
    end

    return max_ratio   # = ρ(Ly, D_bound, β), the per-layer rate (NOT the full-lattice C)
end


# =====================================================================
# Per-config row weights  (the analytical diagnostic — see ANALYTICAL_HOWTO.md)
# =====================================================================
# MOTIVATION: compute_rho_plateau returns only max_s a(s)/a_D(s).  For the
# analytical work we need the FULL per-config weight arrays at the saturated
# top row: a(s) = ⟨v(s)|β⟩ (exact) and a_D(s) = ⟨v(s)|β_D⟩ (truncated), for
# every one of the 2^Ly single-row spin configs s.  From these we build
# g(s) = a_D(s)/a(s), the exact conditionals p(s) = a(s)/Σa, and hence the
# closed-form per-row ratio r(s) = E_p[g]/g(s).
#
# Same environment machinery as compute_rho_plateau, but returns (pe, pt)
# — the raw (environment-normalised) weight arrays, length 2^Ly each.
# The overall scale of pe and pt is irrelevant: it cancels in every object
# we build from them (conditionals, g up to a constant, spreads).

function perrow_weights(Lx::Int, Ly::Int, beta::Float64, D_bound::Int; J::Float64=1.0)
    A, s_inds, v_inds = create_ising_peps(Lx, Ly, beta, J)
    b = [Index(2, "Site,b=$y") for y in 1:Ly]
    # bottom row → initial environment
    tLx = [replaceinds(A[Lx,y]*ITensor([1.0,1.0], s_inds[Lx,y]), [v_inds[Lx-1,y]]=>[b[y]]) for y in 1:Ly]
    E_exact = MPS(tLx); E_trunc = MPS(tLx)
    # sweep up to saturate both environments
    for x in Lx-1:-1:2
        tmpo = [replaceinds(A[x,y]*ITensor([1.0,1.0], s_inds[x,y]),
                            [v_inds[x,y], v_inds[x-1,y]] => [b[y], b[y]']) for y in 1:Ly]
        MPOx = MPO(tmpo)
        E_exact = noprime(apply(MPOx, E_exact; maxdim=2^Ly, cutoff=1e-15)); orthogonalize!(E_exact,1); E_exact[1] ./= norm(E_exact[1])
        E_trunc = noprime(apply(MPOx, E_trunc; maxdim=D_bound, cutoff=1e-10)); orthogonalize!(E_trunc,1); E_trunc[1] ./= norm(E_trunc[1])
    end
    # measure every row-1 config against both environments
    pe = zeros(2^Ly); pt = zeros(2^Ly)
    for i in 0:(2^Ly-1)
        spins = [(i>>(j-1))&1 for j in 1:Ly]
        rte = ITensor[]; rtt = ITensor[]
        for y in 1:Ly
            T = A[1,y]*ITensor(spins[y]==0 ? [1.0,0.0] : [0.0,1.0], s_inds[1,y])
            push!(rte, T*replaceinds(E_exact[y], b[y]=>v_inds[1,y]))
            push!(rtt, T*replaceinds(E_trunc[y], b[y]=>v_inds[1,y]))
        end
        pe[i+1] = abs(scalar(reduce(*, rte)))
        pt[i+1] = abs(scalar(reduce(*, rtt)))
    end
    return pe, pt   # exact and truncated per-config row weights (2^Ly each)
end


# =====================================================================
# ρ tracker: per-row ρ_x and fidelity vs depth (relabel of run_error_tracker)
# =====================================================================
# MOTIVATION: While compute_rho_plateau returns only the saturated
# plateau value, this function tracks ρ_x at EVERY row during the
# bottom→top sweep.  This shows how the boundary environment converges
# to its fixed point: ρ_x should plateau after enough rows, confirming
# that the per-layer rate is well-defined.
#
# Also records fidelity |⟨E_exact|E_trunc⟩| at each step — a scalar
# measure of how similar the two boundary MPS are.
#
# NOTE on the [1,1] cap: for interior rows (x > 1), the measurement
# row's upper vertical bond is closed with ITensor([1,1], v_inds[x-2,y]),
# which MARGINALISES over everything above the row.  This makes ρ_x a
# marginal quantity (see CLAUDE.md §9.2).  compute_rho_plateau avoids
# this by measuring at the genuine top row (no upper bond).

function rho_tracker(Lx::Int, beta::Float64, D_bound::Int; Ly::Int=8, J::Float64=1.0)
    A, s_inds, v_inds = create_ising_peps(Lx, Ly, beta, J)
    b = [Index(2, "Site,b=$y") for y in 1:Ly]

    # Initialise from bottom row (same as compute_rho_plateau)
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
        # Build transfer MPO for row x (same as compute_rho_plateau)
        tensors_mpo = ITensor[]
        for y in 1:Ly
            T = A[x, y] * ITensor([1.0, 1.0], s_inds[x, y])
            T = replaceinds(T, [v_inds[x, y], v_inds[x-1, y]] => [b[y], b[y]'])
            push!(tensors_mpo, T)
        end
        MPOx = MPO(tensors_mpo)

        # Advance exact environment (lossless)
        E_exact = noprime(apply(MPOx, E_exact; maxdim=exact_maxdim, cutoff=1e-16))
        orthogonalize!(E_exact, 1); ne = norm(E_exact[1]); E_exact[1] ./= ne; logZe += log(ne)
        # Advance truncated environment (lossy compression)
        E_trunc = noprime(apply(MPOx, E_trunc; maxdim=D_bound, cutoff=1e-10))
        orthogonalize!(E_trunc, 1); nt = norm(E_trunc[1]); E_trunc[1] ./= nt; logZt += log(nt)

        # Fidelity = overlap between exact and truncated environments (0 to 1)
        push!(fidelity, abs(inner(E_exact, E_trunc)))

        # Brute-force ρ_x at this row
        mr = 0.0
        for i in 0:(2^Ly - 1)
            spins = [(i >> (j - 1)) & 1 for j in 1:Ly]
            rte = ITensor[]; rtt = ITensor[]
            for y in 1:Ly
                T = A[x-1, y]
                sv = spins[y] == 0 ? [1.0, 0.0] : [0.0, 1.0]
                T *= ITensor(sv, s_inds[x-1, y])
                # For interior rows, close the upper vertical bond with [1,1]
                # (marginalises over spins above — this is the [1,1] cap of §9.2)
                if (x - 1) > 1
                    T *= ITensor([1.0, 1.0], v_inds[x-2, y])
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
        push!(rows, Lx - x + 1)   # "rows from bottom" (1 = second-to-last row)
        push!(rho_x, mr)
    end

    return rows, fidelity, rho_x
end


# =====================================================================
# Saturated boundary MPS (convenience accessor)
# =====================================================================
# MOTIVATION: Some diagnostics (like truncation_error_fidelity) need
# the boundary MPS after it has converged to its fixed point.  This
# returns the environment at row 2 (just below the top), which is the
# saturated boundary MPS after Lx-1 transfer-matrix applications.

saturated_boundary_mps(Lx::Int, Ly::Int, beta::Float64, D_bound::Int; J::Float64=1.0) =
    compute_bottom_envs(create_ising_peps(Lx, Ly, beta, J)..., Lx, Ly, D_bound)[2]


# =====================================================================
# Truncation error ε_D between exact and D-truncated boundary MPS
# =====================================================================
# MOTIVATION: The discarded Schmidt weight ε_D quantifies how much
# information is lost by truncating the boundary MPS to bond dim D.
# We compute it as  ε_D = √(1 − |⟨e|e_D⟩|²)  where e is the exact
# (D=2^Ly) and e_D the truncated boundary MPS.  This is related to ρ
# (Phase 2.4): larger truncation error → larger per-layer mixing penalty.
# Crucially, ε_D is measurable without brute-force enumeration, making
# it a practical diagnostic for proposal quality.

function truncation_error_fidelity(Lx::Int, Ly::Int, beta::Float64, D::Int; J::Float64=1.0)
    e  = saturated_boundary_mps(Lx, Ly, beta, 2^Ly; J=J)   # exact (full bond dim)
    eD = saturated_boundary_mps(Lx, Ly, beta, D;     J=J)   # truncated to D
    ov = abs(inner(e, eD)) / (norm(e) * norm(eD))           # normalised overlap
    return sqrt(max(0.0, 1 - ov^2))  # ε_D = sine of the "angle" between e and e_D
end


# #####################################################################
#         PHASE 3 — AVERAGE-CASE / TYPICAL MIXING DIAGNOSTICS
# #####################################################################
# MOTIVATION: Phase 2 shows that the worst-case C = ρ^Lx is exponential
# in volume at β_c.  But empirically, TNMH has high acceptance rates and
# fast equilibration even at criticality.  The resolution: the worst case
# is realised by an exponentially rare configuration, while the TYPICAL
# importance weight w(x) is close to 1.
#
# The average-case mismatch is captured by:
#   χ²(π‖q) = E_q[w²] − 1 = Var_q(w)
# which stays mild even when C = max w is large, because C is set by the
# tail of the weight distribution, not its bulk.
#
# We also measure the restricted constant C_S (max w over a high-probability
# set S), and the empirical integrated autocorrelation time τ_int of the
# real TNMH chain.  Together these explain why practice is fast.
# #####################################################################


# =====================================================================
# Draw samples from q and compute self-normalised importance weights
# =====================================================================
# MOTIVATION: For large lattices where enumeration is impossible, we
# can SAMPLE from q (using sample_config_opt) and estimate the weight
# distribution.  The key trick: we don't need to know Z a priori.
# Since  E_q[w] = E_q[π/q] = Σ_x q(x)·π(x)/q(x) = 1  (self-normalised),
# we estimate  log Z  from the sample and then compute  log w(x) = log π(x) − log q(x).

function weight_samples(Lx::Int, Ly::Int, beta::Float64, D_bound::Int, n::Int; J::Float64=1.0)
    E = zeros(Float64, n); logq = zeros(Float64, n)
    for i in 1:n
        # Draw a random config from the proposal q and record its log q and energy
        cfg, lq = sample_config_opt(Lx, Ly, beta, D_bound, J)
        E[i] = measure_energy(cfg, J)
        logq[i] = lq
    end

    # Estimate Z from the sample:  Z = E_q[e^{-βE}/q] ≈ (1/n) Σ_i e^{-βE_i}/q(x_i)
    # In log space:  log Z ≈ logsumexp(-β·E - logq) - log(n)
    logZ_est = logsumexp(-beta .* E .- logq) - log(n)

    # Self-normalised log-weights:  log w = log π − log q = (-βE − log Z) − log q
    logw = (-beta .* E .- logZ_est) .- logq

    return (E=E, logq=logq, logw=logw, logZ_est=logZ_est)
end


# =====================================================================
# χ²(π‖q) — exact (from enumeration output)
# =====================================================================
# MOTIVATION:  χ²(π‖q) = Σ_x q(x) · w(x)² − 1 = E_q[w²] − 1.
# This is the variance of the importance weights under q and controls
# the effective sample size of importance sampling.  We compute it
# exactly from the enumerate_weights output (small lattices only).

chi2_exact(res) = sum(exp.(res.logq .+ 2 .* res.logw)) - 1.0


# =====================================================================
# χ² — estimated from samples, with reliability diagnostics
# =====================================================================
# MOTIVATION: For larger lattices, estimate χ² from sampled weights.
# Also returns ESS (effective sample size = (Σw)²/Σw², tells how many
# independent samples the weighted sample is worth) and the largest
# observed weight (a lower bound on C).

function chi2_sampled(logw::AbstractVector)
    w = exp.(logw)
    return (chi2 = mean(w .^ 2) - 1.0,          # estimated χ²
            ess  = sum(w)^2 / sum(w .^ 2),       # effective sample size
            wmax = maximum(w),                     # largest observed weight
            n    = length(w))                      # number of samples
end


# =====================================================================
# Restricted constant C_S = max_{x ∈ S} w(x)
# =====================================================================
# MOTIVATION: C = max over ALL configs is dominated by an exponentially
# rare worst-case config.  If we restrict the max to a high-probability
# set S (e.g. configs within an energy band around ⟨E⟩_π), we get
# C_S ≪ C.  This is the basis of a "restricted conductance" argument:
# the chain mixes fast within S, and rarely visits S^c.

restricted_C(logw::AbstractVector, mask::AbstractVector{Bool}) =
    any(mask) ? exp(maximum(logw[mask])) : NaN


# =====================================================================
# Integrated autocorrelation time τ_int (Sokal's windowed estimator)
# =====================================================================
# MOTIVATION: τ_int measures how many MCMC steps are needed to get one
# independent sample.  For an observable f (here, energy), the variance
# of the sample mean is  Var(f̄) = Var(f) · 2τ_int / n.  So τ_int
# quantifies the "cost" of correlation in the chain.
#
# Sokal's windowed estimator sums the autocorrelation function C(t)
# up to a self-consistent window  t_max = c · τ_int (here c = 5).
# This avoids summing noise from large lags where C(t) ≈ 0 but has
# large statistical fluctuations.

function integrated_autocorr(series::AbstractVector; c::Float64=5.0)
    x = series .- mean(series)             # centre the series (subtract mean)
    n = length(x)
    v0 = sum(x .^ 2) / n                  # variance estimate (lag-0 autocorrelation)
    v0 == 0 && return 0.5                  # constant series → no correlation

    tau = 0.5                              # initialise: τ = 0.5 (the lag-0 contribution)
    for t in 1:(n - 1)
        # Normalised autocorrelation at lag t:
        #   C(t) = (1/(n−t)) Σ_{i=1}^{n−t} x_i · x_{i+t}  / v0
        ct = (sum(@views x[1:n-t] .* x[t+1:n]) / (n - t)) / v0
        tau += ct

        # Sokal's automatic windowing: stop summing when the window exceeds
        # c × current τ estimate.  Beyond this point, the autocorrelation
        # is mostly noise and adding it would increase the estimator's variance.
        t > c * tau && break
    end
    return tau    # units = number of chain steps per independent sample
end


# #####################################################################
#                PHASE 4 — INDEPENDENT VALIDATIONS
# #####################################################################
# MOTIVATION: Cross-check the C measurement by two independent methods:
# (1) CFTP coalescence time (gives C through a completely different route)
# (2) Comparison with single-spin-flip Glauber dynamics (quantifies the
#     speedup from collective TN updates)
# #####################################################################


# =====================================================================
# CFTP (Coupling From The Past) coalescence time
# =====================================================================
# MOTIVATION: For an IMH chain, Coupling From The Past has a simple
# structure: all coupled chains (started from every possible state)
# coalesce the FIRST time the worst-case configuration s★ = argmax w(x)
# is proposed.  This is because s★ has the highest importance weight, so
# it is ALWAYS accepted regardless of the current state — and once all
# chains are in the same state, they stay coupled forever.
#
# The probability of proposing s★ in any given step is q(s★), so the
# mean coalescence time is  1/q(s★).  Since  w(s★) = C  and
# q(s★) = π(s★)/C, this gives  mean time = C/π(s★).
#
# By measuring the empirical coalescence time and comparing to the
# predicted 1/q(s★), we get an INDEPENDENT re-derivation of C.

function cftp_coalescence(Lx::Int, Ly::Int, beta::Float64, D_bound::Int;
                          ntrials::Int=200, maxsteps::Int=10^6, J::Float64=1.0)
    # First, enumerate to find s★ (the worst-case config) and q(s★)
    res = enumerate_weights(Lx, Ly, beta, D_bound; J=J)
    # Convert 1-based argmax index back to a config matrix
    # (argmax is 1-based, config_from_int expects 0-based integer)
    star_cfg = config_from_int(res.argmax - 1, Lx, Ly)

    times = Int[]
    for _ in 1:ntrials
        steps = 0
        while steps < maxsteps
            steps += 1
            # Draw a proposal from q
            cfg, _ = sample_config_opt(Lx, Ly, beta, D_bound, J)
            # Check if this proposal IS the worst-case config s★
            cfg == star_cfg && break
        end
        push!(times, steps)
    end

    qstar = exp(res.logq[res.argmax])   # q(s★)
    return (mean_time = mean(times),    # empirical mean coalescence time
            predicted = 1 / qstar,      # predicted = 1/q(s★)
            C = res.C,                  # Mengersen–Tweedie constant for reference
            times = times)              # full distribution of coalescence times
end


# =====================================================================
# Single-spin-flip Glauber dynamics (reference baseline)
# =====================================================================
# MOTIVATION: The whole point of TNMH is to replace slow local dynamics
# (single-spin-flip Metropolis / Glauber) with fast collective updates.
# To quantify the speedup, we implement a standard Glauber chain and
# measure its integrated autocorrelation time τ_int.  Comparing
# τ_int(TNMH) vs τ_int(Glauber) gives the speedup factor.
#
# One "sample" = one full lattice sweep of Lx·Ly attempted single-spin
# flips (so both methods are compared per "sweep", not per flip).
#
# The Glauber/Metropolis rule for flipping spin (x,y):
#   ΔE = 2 J s_{x,y} Σ_{neighbours} s_{nb}
#   Accept flip if  ΔE ≤ 0  (energetically favourable)
#   or with probability  exp(−β ΔE)  otherwise.

function glauber_tau_int(Lx::Int, Ly::Int, beta::Float64, N_samples::Int;
                         J::Float64=1.0, burnin_frac::Float64=0.2)
    # Random initial configuration in {-1, +1}
    spins = rand((-1, 1), Lx, Ly)

    # Total Ising energy:  H = -J Σ_{⟨ij⟩} s_i s_j  (sum over nearest-neighbour pairs)
    energy(s) = -J * (sum(s[:, 1:end-1] .* s[:, 2:end]) +    # horizontal bonds
                      sum(s[1:end-1, :] .* s[2:end, :]))       # vertical bonds

    # Energy change from flipping spin (x,y):
    #   ΔE = 2 J s_{x,y} · (sum of neighbours)
    # This is O(1) per flip (only 4 neighbours to check), vs O(N) for a full energy.
    function dE(s, x, y)
        nb = 0
        x > 1  && (nb += s[x-1, y])    # neighbour above
        x < Lx && (nb += s[x+1, y])    # neighbour below
        y > 1  && (nb += s[x, y-1])    # neighbour left
        y < Ly && (nb += s[x, y+1])    # neighbour right
        return 2 * J * s[x, y] * nb
    end

    Es = Float64[]
    for _ in 1:N_samples
        # One "sweep" = Lx·Ly random single-spin flip attempts
        for _ in 1:(Lx * Ly)
            x = rand(1:Lx); y = rand(1:Ly)           # pick a random site
            d = dE(spins, x, y)                        # energy change if flipped
            # Metropolis acceptance:  accept if ΔE ≤ 0, else with prob e^{-βΔE}
            (d <= 0 || rand() < exp(-beta * d)) && (spins[x, y] *= -1)
        end
        push!(Es, energy(spins))   # record energy after this sweep
    end

    # Discard burn-in period (first 20% by default) before computing τ_int
    burn = Int(floor(burnin_frac * N_samples))
    return (tau_int = integrated_autocorr(Es[burn+1:end]), E = Es)
end
