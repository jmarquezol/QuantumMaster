# CLAUDE.md — Master Thesis Project Context
# Out-of-Equilibrium Dynamics of an ANNNI-type Model
# Author: Joaquín G. Márquez Olguín
# Supervisor: Stefano Carignano (BSC)
# Period: February–September 2026

>>> NEW SESSION: read CONTINUATION.md FIRST — it is the latest handoff (session state, current
>>> results, and the prioritized next-steps plan: make the block PM cheaper, then fix NB7 p=0.1).

---

## 1. PROJECT GOAL

Study the out-of-equilibrium dynamics of an ANNNI-type model, a non-integrable self-dual
quantum spin chain with next-to-nearest-neighbour (NNN) interactions. The method is
transverse contraction via ITransverse.jl. The core observable is the Loschmidt echo
(return amplitude after a quench), from which we extract generalized temporal entropies
(Rényi-2) along the temporal MPS.

Central research question: does the model at its critical point display the same
logarithmic temporal entropy growth and emerging dual-unitarity as the pure Ising model?
If yes, CFT universality class (c=1/2, Ising) survives NNN frustration in the temporal
direction as well as the spatial one.

Confirmed from DMRG: at p <= ~2, the model is in the Ising universality class (c ≈ 0.5),
so we expect the temporal entropies to match the Ising CFT prediction at small p.

---

## 2. THE ANNNI-type HAMILTONIAN (EXACT FORM — SIGN CONVENTIONS ARE CRITICAL)

H = -sum_i [ sigma_z_i * sigma_z_{i+1}
           + p * sigma_z_i * sigma_z_{i+2}
           + lambda * sigma_x_i
           + p * lambda * sigma_x_i * sigma_x_{i+1} ]

Parameters:
- sigma_x, sigma_z: standard Pauli matrices (spin-1/2, d=2)
- lambda: transverse field coupling (critical point at lambda=1 for all studied p)
- p: NNN coupling strength (p=0 exactly recovers the standard integrable TFIM)
- Self-duality condition: H(lambda, p) = lambda * H(1/lambda, p)
  The pλ σx_i σx_{i+1} term is REQUIRED to maintain self-duality; without it the
  dual image of the NNN Ising term would produce a new coupling not present in H.

CRITICAL CONVENTION MISMATCH WITH ITRANSVERSE BUILT-IN ISING:
The built-in Ising model in ITransverse.jl uses:
  H_ITransverse = -sum_i [ J * sigma_x_i * sigma_x_{i+1} + g * sigma_z_i + h * sigma_x_i ]
  (coupling on sigma_x, field on sigma_z)

The ANNNI-type model uses:
  (coupling on sigma_z, field on sigma_x — OPPOSITE basis)

These are related by a pi/2 rotation: sigma_x <-> sigma_z.
Consequence: the initial state |psi_0> and any operator expectations must be defined
consistently with whichever convention the MPO is using.
For the custom Alcaraz MPO from ITensorExpMPOv2.jl, the convention is sigma_z coupling.
The TDVP benchmark must use exactly the same sign form and initial state.

CONFIRMED INITIAL STATE: |psi_0> = |X+> (the +1 eigenstate of sigma_x, i.e. [1,1]/sqrt(2)
in the Z basis), NOT |Z-up> as an earlier draft of this file guessed. This is consistent
across ALL code paths in src/models.jl and every notebook:
  - TDVP / Schrodinger:  psi0 = complex(MPS(sites, "X+"))
  - Transverse:          init_state = complex(state(s, "X+")), set as tp.bl AND passed
                         as tr= to fw_tMPO / fw_tMPS.
|X+> is the ground state of H at lambda -> infinity (the polarized paramagnet in this
sigma_x-field convention). A sign/basis mismatch here is the classic source of a wrong
Loschmidt echo, so any new code MUST keep using "X+".

BOUNDARY CONDITION MAPPING (verified June 2026, now in nb7 / legacy/18 Part 3):
  |X+> = field-polarized paramagnet = DISORDERED in the order parameter σ_z
       = analog of paper's free BC |↑⟩ (σ_z eigenstate in their σ_z-field convention)
       = FREE BC, x₁ = 1/2, FAST convergence to CFT predictions
  |Z±> = σ_z eigenstate = ORDERED in the order parameter
       = analog of paper's fixed BC |+⟩ = FIXED BC, x₁ = 2, SLOW convergence
  Convention swap: our field is on σ_x, paper's field is on σ_z → σ_x↔σ_z.
  CONFIRMED: Eq.(4) gives x₁(p=0)=0.502, x₁(p=0.1)=0.497, a₁=0.788≈π/4 (free BC exactly).

---

## 3. FSM MPO REPRESENTATION OF H_Alcaraz (bond dimension D_w = 5)

The Hamiltonian is encoded as a finite-state machine MPO. Local tensor W_Alcaraz is a
5x5 upper-triangular block matrix:

W_Alcaraz =
[ I      -sigma_z   -p*lam*sigma_x   -p*sigma_z   -lam*sigma_x ]
[ 0       0          0                0             sigma_z      ]
[ 0       0          0                0             sigma_x      ]
[ 0       I          0                0             0            ]
[ 0       0          0                0             I            ]

Block identification:
- Row 1 (start): initial state
- Col 5 (end): final absorbing state
- (1,2): initiates NN z-coupling (sigma_z at site i)
- (1,3): initiates NN x-coupling (-p*lam*sigma_x at site i)
- (1,4): initiates NNN z-coupling (-p*sigma_z at site i), bridges i -> i+2
- (1,5): local field (-lam*sigma_x), immediate
- (2,5): terminates NN z-coupling (sigma_z at site i+1)
- (3,5): terminates NN x-coupling (sigma_x at site i+1)
- (4,2): NNN bridge — applies I at intermediate site i+1, carries z-coupling forward
- (5,5): identity propagation

The upper-triangular structure means the FSM reads LEFT-TO-RIGHT only.
This makes the MPO INHERENTLY ASYMMETRIC (not left-right symmetric).
This has consequences for which ITransverse power method to use (see Section 7).

Built via ITensorExpMPOv2.jl which handles the exponentiation of this FSM (Section 5b).
We use Van Damme 2nd-order expressions (VD2; SciPost Phys. 17, 135, 2024) for U(dt).
NOTE: in practice we do NOT hand-build the 5x5 W above — `expmpo` derives the FSM
automatically from the OpSum (alcaraz_opsum), so the table is documentation, not code.

---

## 3b. MODEL DEFINITIONS IN src/models.jl (two models, one shared pattern)

src/models.jl (was main.jl) defines TWO models, each with the SAME boilerplate so they plug into
both the Schrodinger (TDVP / direct MPO) and the transverse (ITransverse) pipelines:

  (a) AlcarazParams      — the real target model. alcaraz_opsum(N, lambda, p):
        -ZZ(i,i+1)  - p*lambda*XX(i,i+1)  - p*ZZ(i,i+2)  - lambda*X(i).
        Self-dual; the XX term is present (Section 2).
  (b) TricriticalParams  — tricritical Ising with 3-body ZZX / XZZ terms (an OPTIONAL variant kept
        for future exploration; no notebook in the current series). tricritical_opsum(N, lambda).

  DROPPED: the old BenchmarkParams (Alcaraz with the XX term off — a NNN TFIM) used only to probe
  the now-resolved transverse-vs-TDVP "discrepancy"/DQPT thread. It no longer exists in code; its
  only salvage is the MPO-exp benchmark (→ nb2) and the block-PM intro (→ nb5).

Shared pattern for each model XXX:
  - mutable struct XXXParams <: ModelParams  with fields lambda, (p,) phys_site::Index.
  - abstract type AbstractXXXRecipe <: ExpHRecipe  with structs XXXWI / XXXWII / XXXVD2,
    and _alg_string(::XXXVD2) = "VD2" etc.
  - xxx_opsum(N, ...) :: OpSum                          (the Hamiltonian)
  - expH_xxx(sites, ...; dt, mpo_alg="VD2")             (direct Schrodinger MPO via expmpo)
  - ITransverse.expH(sites, mp::XXXParams, recipe::AbstractXXXRecipe; dt)  <-- THE HOOK
        ITransverse calls this to get the spatial U(dt) MPO it then rotates into the tMPO.
        It just returns expmpo(xxx_opsum(...), sites, -im*dt; alg=Algorithm(_alg_string(recipe))).
  Driver helper: the UNIFIED `compute_entropies(mp::ModelParams, T; scheme, nbeta, ...)` in
  src/transverse_tools.jl wires all of the above to powermethod_lr + gen_renyi2 (it replaces the
  old per-model compute_alcaraz/tricritical_entropies). All use init_state = |X+>.

---

## 4. FILE STRUCTURE  (reorganized June 2026 — clean 1–7 notebook series)

master_thesis/
  CLAUDE.md                      <- this file (read first every session)
  README.md                     <- human-facing project overview
  carignano-tagliacozzo.md      <- Carignano & Tagliacozzo 2024 paper in markdown (primary ref)

  src/                           <- CONSOLIDATED LIBRARY (notebooks `include("src/thesislib.jl")`)
    thesislib.jl                   <- entry point: all `using` + includes models.jl & transverse_tools.jl
    models.jl                      <- model defs: AlcarazParams / TricriticalParams, opsums, expmpo
                                      wrappers, ITransverse.expH dispatch (was main.jl). NOTE: the
                                      old Benchmark model — Alcaraz minus the XX term — was DROPPED.
    transverse_tools.jl            <- build_alcaraz_tmpo, block_transfer_eigs, lincomb_mps,
                                      run_pm_diagnosed, compute_entropies (UNIFIED, model-agnostic;
                                      replaces compute_alcaraz/tricritical_entropies; supports
                                      use_block_pm=true for robust block-PM eigenvector extraction),
                                      plot_entropy_profiles (same use_block_pm passthrough),
                                      tdvp_loschmidt_amplitude, crashsafe_sweep, plot_panels.
                                      Z2 helpers (z2_operator/project_parity) DROPPED. Caches default
                                      to results/data/, figures to results/imgs/. The old `./=`
                                      norm bug is fixed (scalar mult, see §10 gotcha).

  ITensorExpMPOv2.jl/            <- exp-MPO package — a FORK of tipfom/ITensorExpMPO.jl
                                    (github.com/tipfom/ITensorExpMPO.jl). ALL code is upstream (@tipfom)
                                    EXCEPT the VD2 kernel in src/eulerbuilder.jl, which is
                                    this thesis's only original addition (credit it as such).
    src/
      ITensorExpMPO.jl           <- module file; exports `expmpo`
      opsum_to_U_generic.jl      <- expmpo() entry point (dispatches qn vs non-qn)
      opsum_to_U.jl              <- finitestatemachine() (FSM build + SVD channel compression)
                                    and eulermpo() (assembles the MPO, calls makeW per site)
      eulerbuilder.jl            <- makeW kernels: WI, WII (Zaletel), VD2 (Van Damme 2nd order — OURS)
      opsum_to_U_qn.jl           <- QN-conserving variant (not used in current runs)
  ITransverse_source/            <- CLONED SOURCE of ITransverse.jl (NEWER than the installed version;
    src/                            read-only API reference — verify signatures here, see Section 9)

  NOTEBOOKS — the clean narrative series (built June 2026 from the best parts of the old ones):
    1_introduction_model.ipynb     <- model + naive Trotter TEBD (3-site NNN gates) benchmarked
                                      against cached TDVP ⟨Z⟩ (visible mismatch → motivates nb2);
                                      TDVP Loschmidt rate + entanglement barrier.
    2_mpo_exponentiation.ipynb     <- FSM exp-MPO: WI/WII/VD2 + Schrödinger benchmark vs TDVP;
                                      WII works better but breaks past some T (investigated later)
    3_temporal_entropies.ipynb     <- generalized temporal (Rényi-2) entropies from the tMPS;
                                      single-vector PM stops converging at gap closing (T≳5) →
                                      block PM (use_block_pm=true) recovers the conformal dome;
                                      head-to-head v1 vs block PM at T=5.5; VD2 vs WII comparison.
                                      APPENDIX (additive): dt-dependence of the barrier onset T_crit
                                      (dt=0.05 nbeta=8 vs dt=0.1 nbeta=4), single-vector then block PM.
    3.5_block_pm_efficiency.ipynb  <- making block_transfer_eigs cheaper: profile the bottleneck
                                      (2k applyn + 2k lincomb per iter; converged χ ≪ maxdim cap),
                                      then benchmark routes A maxdim / B maxdims-ramp / C cutoff /
                                      D WII-vs-VD2 / E warm-start (seedL/seedR) / F k. Reference
                                      point p=0.1,T=6,nbeta=4; results cached to nb35_blockpm_bench.
    4_cft_ground_state.ipynb       <- DMRG equilibrium central charge: c(p) sweep, finite-size
                                      scaling S(L/2) vs ln(L), full chord fit S(x) for p=0/0.1.
                                      Three independent reads all give c≈1/2 (KEY confirmed result).
    5_spectral_gap_degeneracy.ipynb<- deflation (single-vector) fails (NaN/Inf) → BLOCK power method;
                                      gap closes FASTER at larger p (barrier onset earlier);
                                      DQPTs introduced for completeness (Ising has one; our quench
                                      TO criticality does NOT).
    6_loschmidt_ising.ipynb        <- reproduce Carignano-Tagliacozzo for the integrable Ising chain
                                      (symmetric Murg + powermethod_sym): c=1/2 recovered. WORKS.
                                      Dominant eigenvalue λ₀(T) traced in the complex plane — the
                                      "circle plot" confirming emergent dual unitarity.
    7_loschmidt_alcaraz.ipynb      <- REMADE (June 2026): Alcaraz temporal central charge via the
                                      well-converged block PM (itermax=8000, nbeta=4), TWO routes —
                                      Rényi-2 entropy slope (c=8·slope) + leading eigenvalues
                                      (Eq.3 c, Eq.4 x1, λ0 circle) — with p=0 as the convention
                                      calibration (headline = p=0.1/p=0 ratio). Single master
                                      block-PM sweep → results/data/nb7_alcaraz_block.jld2.
                                      OUTCOME PENDING RE-RUN (the old c≈0.69→6.0 was from the
                                      under-converged itermax=800 block PM — superseded).
    TN_assignment.ipynb            <- course exercise (Ising quench TEBD/transverse); not thesis work

  results/
    imgs/                          <- figures (cft_L, p_dependence, cft_ising_validation,
                                      block_pm_ising_p0.0, block_pm_ising_vs_alcaraz,
                                      ising_lambda0_circle)
    data/                          <- cached .jld2:
      block_pm_alcaraz_p0.0.jld2   <- block PM sweep p=0   (T=0.5..7.0, nbeta=0) — eigenvalues
      block_pm_alcaraz_p0.1.jld2   <- block PM sweep p=0.1 (T=0.5..5.5, nbeta=0) — eigenvalues
      cft_renyi2_beta_p5.jld2      <- Alcaraz Rényi-2 c sweep (nbeta=4) — the "bad c" data for nb7
      tdvp_loschmidt_p0.1_N40.jld2 <- TDVP Loschmidt amplitude N=40 (T=0.5..7.0)
      tdvp_loschmidt_p0.1_N80.jld2 <- TDVP Loschmidt amplitude N=80 (T=0.5..4.0)
      tvdp_run.jld2                <- cached TDVP ⟨Z⟩ benchmark (p=0.5, N=50, |Up⟩) for nb1
      ising_lambda0.jld2           <- symmetric Ising λ₀(T) sweep for nb6 circle plot
      nb7_alcaraz_block.jld2       <- (created on run) NB7 master block-PM sweep: per-(p,T)
                                      leading eigenvalues + Rényi-2 profile, nbeta=4, itermax=8000.
                                      Supersedes cft_renyi2_beta_p5 + block_pm_alcaraz_p0.{0,1}.
      nb7_alcaraz_lite.jld2        <- (created on run) NB7 LITE Route-2 sweep: eigenvalues only via
                                      block_transfer_eigs(k=2, itermax=400) — cheap λ0,λ1 per (p,T).
      nb35_blockpm_bench.jld2      <- (created on run) NB3.5 block-PM cost benchmark, keyed by config.
      rate_*.jld2                  <- MPO-exp / rate benchmarks (WII, VD2, TDVP) for nb1/nb2

  legacy/                        <- ARCHIVE (git-recoverable; safe to purge once satisfied):
                                    the old numbered notebooks (1,2,5,6,7,8,9,10,13,16,17,18), the
                                    original main.jl & dqpt_diagnostics.jl, and obsolete Z2/DQPT/
                                    benchmark/overlap data & figures from the closed investigations.

---

## 5. PACKAGE ECOSYSTEM

ITransverse.jl
  Author: Stefano Carignano (BSC)
  URL: https://github.com/starsfordummies/ITransverse.jl
  Purpose: Transverse contraction algorithms (power method, light cone, RTM truncation,
           temporal entropy computation)
  Status: Private package, installed from GitHub (old version).
          Reference source cloned into ITransverse_source/ for reading.
  Note: Package is evolving; function names and signatures may differ between the
        installed version and HEAD. Always verify against ITransverse_source/src/.

ITensorExpMPOv2.jl (our package — see Section 5b for the full algorithm)
  Purpose: OpSum (Hamiltonian) -> exp(-i H dt) directly in MPO form (a single time-step
           propagator U(dt) as an MPO), via a finite-state-machine + Euler-style builder.
  Public API: ONE exported function:
      expmpo(os::OpSum, sites, tau; alg=Algorithm"WII"(), mindim=1, maxdim=typemax(Int),
             cutoff=1e-15)  ->  MPO
    where tau = -im*dt for real-time evolution (or tau = -dt for imaginary time).
    `alg` selects the exponentiation kernel: Algorithm("WI"), ("WII"), or ("VD2").
  In our code expmpo is always called with tau = -im*dt; the imaginary-time blocks the
  transverse method needs are built by ITransverse separately using dbeta = -im*dt.
  Status: Treated as stable. The Schrodinger benchmark (TDVP vs direct VD2/WII MPO apply)
          AGREES, which validates the MPO construction and the Trotter order — so when a
          transverse-vs-TDVP discrepancy appears, the MPO is NOT the prime suspect.
  Dev workflow: activate the env in the Julia REPL, then `dev ./ITensorExpMPOv2.jl/` to track
          local edits to the fork.

ITensors.jl / ITensorMPS.jl
  Standard tensor network library for Julia. ITransverse is built on top of it.
  Use SpinHalf site type for spin-1/2 systems.

---

## 5b. HOW expmpo BUILDS U(dt) (ITensorExpMPOv2.jl internals)

expmpo turns an OpSum H into the MPO of exp(tau*H) (tau = -i*dt) in TWO stages:

STAGE 1 — finitestatemachine() in opsum_to_U.jl
  Sweeps the chain left->right and, for each site n, splits every OpSum term that crosses
  the bond into (left, onsite, right) factors, exactly like ITensorMPS's standard
  OpSum->MPO autoMPO. It then SVD-compresses the "memory channels" carried across each
  bond (truncate! with the given maxdim/cutoff), so the FSM uses the minimum bond dim.
  Output per site: four operator blocks in the FSM decomposition
      D = onsite (local) term
      C = term STARTING on site n   (couples to the outgoing/right memory channels)
      B = term ENDING on site n     (couples to the incoming/left memory channels)
      A = term PASSING THROUGH n    (both incoming and outgoing channels)
  These are the standard W = [[D, C],[B, A]] (+ identity track) Hamiltonian-MPO blocks.

STAGE 2 — makeW(alg, ...) in eulerbuilder.jl, called once per site by eulermpo()
  Exponentiates the local W_H block. THREE algorithms (the `alg` argument):

  Algorithm"WI"  — FIRST ORDER.  W_I ~ I + tau*W_H.  Time step spread as
                   tC = sqrt(|tau|), tB = tau/tC across the B and C blocks. Cheapest,
                   O(dt) error. Virtual link dim = 1 + chi.

  Algorithm"WII" — FIRST ORDER (Zaletel et al.; = Van Damme "order-1 MPO", see 5c.7). Maps the
                   four blocks onto two auxiliary "bosons" a, abar (each dim 2), builds the
                   effective local Hamiltonian
                   h = D*tau + B*tB*a^dag + C*tC*abar^dag + A*a^dag*abar^dag, EXPONENTIATES
                   it exactly (exp(h)|0,0>), then projects the result onto the
                   <0,0|,<0,1|,<1,0|,<1,1| boson sectors to read off the W sub-blocks.
                   One-step phase error O(dt^2); globally 1st order for our NNN model (only
                   effectively 2nd order for strictly-NN H). Virtual link dim = 1 + chi.

  Algorithm"VD2" — SECOND ORDER (Van Damme et al., SciPost Phys. 17, 135 (2024); one-step phase
                   error O(dt^3), unconditionally 2nd order — see 5c.7),
                   our preferred scheme. Implements the explicit Appendix-A polynomial
                   expressions (W11..W33) built from symmetric products (sym_sum / otimes)
                   of the tau-scaled A,B,C,D blocks up to 3rd-order symmetric terms.
                   IMPORTANT STRUCTURAL FACT: VD2 EXPANDS the virtual link to
                       dim = 1 + chi + chi^2
                   (vs 1 + chi for WI/WII) — see opsum_to_U.jl:93. makeW recovers the true
                   FSM memory dim chi from the expanded size by solving x^2+x = N_expanded.
                   Consequence for the transverse method: the spatial MPO's virtual bond
                   (= the temporal PHYSICAL dimension after the 90-deg rotation) is LARGER
                   for VD2 than for WI/WII. The benchmark/entropy code reads this dimension
                   dynamically via `dim(inds(b.Wc, "Site,time")[1])` (or dim(linkind(mpo,1)))
                   — never hard-code it.

All three return a plain Array W[ll, rl, s, s'] that eulermpo wraps into U[n]; the chain
ends are capped with the standard L=[...,1] / R=[1,...] boundary vectors.

---

## 5c. WII AND VD2 EXP-MPO CONSTRUCTORS — DETAILED DERIVATION (FOR THE THESIS)

This section is a self-contained, citable explanation of the two second-order exponential-MPO
constructors used in this work, written so it can be pasted to an assistant to help draft the
methods chapter. It describes EXACTLY what ITensorExpMPOv2.jl/src/eulerbuilder.jl computes.
Primary reference for both schemes: Van Damme, Haegeman, McCulloch, Vanderstraeten, "Efficient
higher-order matrix product operators for time evolution", SciPost Phys. 17, 135 (2024)
(WI/WII are originally Zaletel et al., PRB 91, 165112 (2015); VD2 = the WII-style construction
pushed to genuine 2nd order via the Appendix-A polynomials of the Van Damme paper).

### 5c.0 The common starting point: the local Hamiltonian MPO block W_H

A short-range Hamiltonian H = sum of local terms has an EXACT MPO of small bond dimension whose
local tensor is upper-triangular in the virtual ("memory channel") space. After the FSM build +
SVD channel compression (Section 5b, Stage 1), the local block at site n is organised as

        W_H[n]  =  [ D   C ]      (schematically, in the 1 + chi memory basis)
                   [ B   A ]
                   (+ identity track)

with FOUR operator-valued blocks (each a d x d matrix on the physical index, d=2 for spin-1/2):
  D  = on-site term            (scalar channel -> scalar channel)
  C  = term STARTING at n      (scalar -> outgoing memory channels; size 1 x chi_R)
  B  = term ENDING at n        (incoming memory channels -> scalar; size chi_L x 1)
  A  = term PASSING THROUGH n  (incoming -> outgoing; size chi_L x chi_R; carries the "wire")
These are the A,B,C,D arguments handed to makeW(alg, ElT, tau, A, B, C, D). chi (= Nr or Nc in
the code) is the number of compressed memory channels on the left/right bond. For our models
(NN + NNN + field) chi is small (2-3). tau = -i*dt for real time.

A naive exponential exp(tau*W_H) is NOT itself an MPO of the same simple form, because powers of
the upper-triangular W_H mix the blocks (e.g. (tau W_H)^2 produces C·D + D·C, A·D, B·C, ...).
The whole point of WI/WII/VD2 is to repackage exp(tau*W_H) into a NEW local tensor W of a
controlled, small bond dimension with a prescribed local truncation error in tau.

### 5c.1 Time-step splitting convention (identical in all three kernels)

makeW first splits the time step asymmetrically across the starting/ending blocks:
        tC = sqrt(|tau|),    tB = tau / tC      (so tB * tC = tau, |tB| = |tC| = sqrt|tau|)
C (start) is scaled by tC, B (end) by tB, the on-site D by tau, and A by 1. This symmetric
sqrt-splitting is what lets a single auxiliary-boson occupation (WII) or a single polynomial
order (VD2) reproduce the correct cross-terms to O(tau^2): each "leg" of a two-site term carries
sqrt(tau), so a product of a start-leg and an end-leg carries tau, matching exp's quadratic term.

### 5c.2 WI — first order (makeW(::Algorithm"WI", ...))   [context / baseline]

W_I ~ I + tau*W_H, distributed onto the block structure:
  W[1,1]     = I_d + tau*D                       (scalar/identity track, on-site)
  W[1,1+c]   = tC * C_c                          (open an outgoing channel c)
  W[1+r,1]   = tB * B_r                          (close an incoming channel r)
  W[1+r,1+c] = A_{r,c}                            (pass-through wire)
Virtual link dimension = 1 + chi. Local error O(tau^2) => global O(tau) after N sites and many
steps. Cheapest; we use it only as a sanity baseline.

### 5c.3 WII — Van Damme order-1 via two auxiliary bosons (Zaletel construction)

KEY IDEA (the one to explain in the thesis): instead of algebraically squaring W_H and fighting
the cross-terms, embed the four blocks into the dynamics of a tiny fictitious system with TWO
auxiliary hard-core bosons a and abar, each with a 2-dim Fock space {|0>, |1>} (empty/occupied).
Build the effective local Hamiltonian (per pair of memory channels (r,c)):

   h(r,c) = tau * D * Id                         (on-site, no boson)
          + tB * B_r * a^dag                     (ending leg creates boson a)
          + tC * C_c * abar^dag                  (starting leg creates boson abar)
          +      A_{r,c} * a^dag * abar^dag       (pass-through creates BOTH)

Because a, abar are hard-core (a^dag squares to zero on the 2-dim Fock space), exp(h) TRUNCATES
exactly: only occupations 0 or 1 of each boson survive, so the exponential is a finite polynomial
that automatically generates EXACTLY the symmetric cross-terms needed for 2nd-order accuracy
(the B·C, A·D etc. combinations) with NO double counting. We then read off the new W sub-blocks
by exponentiating and projecting onto the four boson sectors:

   w(r,c) = exp(h(r,c)) |0,0>            (apply to the boson vacuum)
   W[1,1]     = <0,0| w   (= exp(tau*D) at leading structure; the identity/scalar track)
   W[1,1+c]   = <0,1| w   (one abar boson  -> outgoing channel: dressed start leg)
   W[1+r,1]   = <1,0| w   (one a boson     -> incoming channel: dressed end leg)
   W[1+r,1+c] = <1,1| w   (both bosons     -> pass-through wire: dressed A)

This is precisely the code: i1, i2 are the two boson Indices; cd1=a^dag, cd2=abar^dag; ket00 the
vacuum; bra00/01/10/11 the projectors; h is assembled then `exp(h) * ket00`; the Array(bra.. * w,
s, s') calls extract each d x d physical block. Edge cases Nr==0 / Nc==0 (no incoming/outgoing
channels, e.g. chain ends or pure on-site) collapse to one boson or to a bare exp(tau*D).
Virtual link dimension = 1 + chi (SAME as WI). In the Van Damme nomenclature WII is the
"first-order MPO" (= Zaletel WII exactly): one-step phase error O(tau^2), so GLOBALLY first
order in general (see 5c.7 for the important NN-vs-NNN subtlety — WII is only effectively 2nd
order for STRICTLY nearest-neighbour H; for our NNN model it is genuinely 1st order).
Cost: one dense matrix exponential of size (2*2*d) x (2*2*d) per memory-channel pair per site.

### 5c.4 VD2 — second order via the explicit Van Damme Appendix-A polynomials

MOTIVATION: WII is only order-1 (and its structure is tied to the hard-core-boson embedding).
The Van Damme paper writes the 2nd-order constructor DIRECTLY as closed-form polynomial
expressions in the (tau-scaled) blocks A,B,C,D, organised into a 3 x 3 BLOCK tensor W (indices
1,2,3 per side) rather than WII's 2 x 2. The "3" comes from allowing each side of the wire to be
in one of three states: scalar (1), a SINGLE active memory leg (2), or a DOUBLED memory leg (3).
Doubling the leg is what buys the clean separation of orders and the larger but more structured
operator. This is our PREFERRED scheme.

Scaled blocks (matching the code):  D~ = tau*D,  C~ = tC*C,  B~ = tB*B,  A~ = A.
Define the EXACT symmetric product sym_sum: for operator-valued matrices X,Y,Z it is the sum
over all DISTINCT orderings of the tensor/operator product (otimes = Kronecker on the virtual
index, matrix product on the physical index). E.g.
   sym_sum(X,X)   = X (x) X
   sym_sum(X,Y)   = X(x)Y + Y(x)X                         (X != Y)
   sym_sum(X,Y,Z) = the 6 permutations X(x)Y(x)Z + ...   (all distinct; with repeats, only the
                    distinct permutations — the code special-cases X===Y, etc.)
sym_sum implements the symmetrized multinomial that the Taylor expansion of exp produces, with
the correct 1/2!, 1/3! weights applied OUTSIDE (the W.. expressions below carry the 1/2, 1/6,
1/3 prefactors).

The nine block entries (eulerbuilder.jl:219-230, verbatim structure; this IS Appendix A of the
Van Damme paper specialised to 2nd order):

   W11 = I + D~ + (1/2) sym_sum(D~,D~) + (1/6) sym_sum(D~,D~,D~)
   W12 = C~ + (1/2) sym_sum(C~,D~) + (1/6) sym_sum(C~,D~,D~)
   W13 = sym_sum(C~,C~) + (1/3) sym_sum(C~,C~,D~)
   W21 = B~ + (1/2) sym_sum(B~,D~) + (1/6) sym_sum(B~,D~,D~)
   W31 = (1/2) sym_sum(B~,B~) + (1/6) sym_sum(B~,B~,D~)
   W22 = A~ + (1/2)[ sym_sum(B~,C~) + sym_sum(A~,D~) ]
            + (1/6)[ sym_sum(C~,B~,D~) + sym_sum(A~,D~,D~) ]
   W23 = sym_sum(A~,C~) + (1/3)[ sym_sum(A~,C~,D~) + sym_sum(C~,C~,B~) ]
   W32 = (1/2) sym_sum(A~,B~) + (1/6)[ sym_sum(A~,B~,D~) + sym_sum(B~,B~,C~) ]
   W33 = sym_sum(A~,A~) + (1/3)[ sym_sum(A~,B~,C~) + sym_sum(A~,A~,D~) ]

Read W11 as "the scalar/identity track carrying just on-site evolution to 3rd Taylor order"; the
12/13 row dresses outgoing legs (single C~, doubled C~C~); 21/31 dresses incoming legs; the 22/23/
32/33 block dresses the through-wire A~ with all its symmetric companions. The single vs doubled
leg (index 2 vs 3) is exactly why VD2 needs a bigger virtual bond than WII.

STRUCTURAL CONSEQUENCE (already in Section 5b, repeated here because it bites the transverse code):
VD2 EXPANDS the virtual link to dim = 1 + chi + chi^2 (opsum_to_U.jl:93-97), vs 1 + chi for
WI/WII. The "+chi^2" is the doubled-leg sector (the W13/W31/W..3/W3. blocks live on a chi^2-sized
index). makeW must RECOVER the true chi from the expanded incoming/outgoing size: the identity
track is stripped before makeW, so ri/ci arrive with dimension N_expanded = chi + chi^2, and the
code inverts x^2 + x = N_expanded  =>  chi = (-1 + sqrt(1 + 4 N_expanded)) / 2 (eulerbuilder.jl:
149-150). After the 90-degree transverse rotation this larger spatial bond becomes a LARGER
temporal physical dimension, so the tMPS site dimension differs between VD2 and WII — ALWAYS read
it dynamically via dim(inds(b.Wc,"Site,time")[1]) (or dim(linkind(mpo,1))); never hard-code it.

One-step phase error O(tau^3) => GLOBALLY second order, i.e. ONE ORDER HIGHER than WII for a
generic (e.g. NNN) Hamiltonian. VD2 is the cheapest scheme that is UNCONDITIONALLY 2nd order
for our next-nearest-neighbour model, which is exactly why we use it for the production
Alcaraz/benchmark runs (see 5c.7 for the verified correspondence with the paper's Appendix A).

### 5c.5 Implementation notes that matter for the write-up / reproducibility

- to_mat_of_mats(T, row_ind, col_ind, ...) turns an ITensor block into a Julia Matrix{Matrix},
  i.e. a chi_L x chi_R grid of d x d physical operators — the data layout on which sym_sum/otimes
  act. otimes(X,Y) = Kronecker product on the virtual (memory) indices, ordinary matrix product
  on the physical d x d operators. place!(mat, r0, c0) drops each sub-block into the final dense
  W_out[1+Nr_expanded, 1+Nc_expanded, d, d], which eulermpo wraps as the site tensor U[n].
- The block decomposition A,B,C,D itself is built in eulermpo (opsum_to_U.jl:106-153) from the
  compressed FSM: the SVD factor Vs[n] rotates the raw memory channels into the minimal set, and
  on-site/start/end/through terms are accumulated into D/C/B/A respectively. So WII and VD2 share
  EXACTLY the same A,B,C,D inputs; they differ ONLY in makeW. This is why switching schemes is a
  one-keyword change (alg=Algorithm("WII") vs Algorithm("VD2")) and why a WII-vs-VD2 comparison
  isolates the exponentiation kernel, nothing else.
- Validation status (Section 10): the Schrodinger benchmark (TDVP vs direct-MPO apply) agrees for
  both WII and VD2, confirming both kernels and the Trotter order are correct.

### 5c.6 What to cite / what is original here

- WI, WII (auxiliary-boson construction): Zaletel, Mong, Karrasch, Moore, Pollmann, PRB 91,
  165112 (2015). The two-boson trick and the <i,j| projection are theirs.
- VD2 (explicit higher-order polynomial blocks): Van Damme, Haegeman, McCulloch, Vanderstraeten,
  SciPost Phys. 17, 135 (2024). The W11..W33 expressions above are their Appendix A specialised to
  second order; our eulerbuilder.jl transcribes them.
- ORIGINAL to this project: the FSM + SVD-channel-compression front-end reused from ITensorMPS's
  autoMPO, the 1+chi+chi^2 link bookkeeping and chi-recovery for VD2, and the integration so the
  SAME U(dt) MPO feeds both the Schrodinger pipeline and the ITransverse transverse rotation.

### 5c.7 RECONCILIATION WITH THE VAN DAMME PAPER (verified term-by-term against Appendix A)

The full paper (arXiv:2302.14181; SciPost Phys. 17, 135 (2024)) is now available. This subsection
records (i) how the paper DERIVES the constructor so the thesis can narrate it, and (ii) a
verified statement that our eulerbuilder.jl VD2 block IS the paper's Appendix-A second-order MPO.

THE PAPER'S CONSTRUCTION LOGIC (Secs 2-6 of the paper; good thesis narrative):
  1. H is a "first-degree" MPO whose local tensor has the upper-triangular block form
     [[I, C, D],[0, A, B],[0,0,I]] (paper Eq. 9). As a finite-state machine (paper Eq. 11) it has
     THREE levels with a clear meaning when read left->right: level 1 = "H has not acted yet",
     level 2 = "H is currently acting" (the through-wire A lives here; bond dim chi), level 3 =
     "H has finished acting". Written out, H = sum_i (D_i + C_i B_{i+1} + C_i A_{i+1} B_{i+2}+...).
  2. GOAL exp(tau H) = sum_n tau^n/n! H^n. Naively summing powers of H is NOT size-extensive:
     ||H^n|Psi>|| scales as N^n (N = system size), so the sum cannot be normalized in the
     thermodynamic limit (paper Eq. 25). The whole construction is about reorganizing this into a
     single size-extensive MPO.
  3. KEY TRICK (paper Sec 3, Alg. 1): represent H^N as a sparse MPO whose levels are TUPLES
     (e.g. (1,1),(1,2),(2,3),(3,3),... for H^2, paper Eq. 22), then FOLD every "finished" level
     (those containing a 3) back onto the start level (1,1...), multiplying by tau^a (N-a)!/N!
     where a = number of 3's folded. This makes the operator size-extensive and automatically
     retains all DISCONNECTED lower-order clusters with the correct exp() prefactors.
  4. EXACT COMPRESSION (paper Sec 4, Alg. 2): equivalent levels are merged by a virtual basis
     rotation (the 1/sqrt(2) matrix, paper Eq. 35), and levels unreachable given the boundary
     vectors L=[1,0,..], R=[1,0,..]^T (paper Eq. 34) are dropped. E.g. (1,2)&(2,1) merge, (2,3)&
     (3,2) merge. These are EXACT (zero singular values), confirmed numerically (paper Sec 7).
  5. EXTENSION (paper Sec 5, Alg. 3): fold in SOME order-(N+1) connected terms at NO extra bond
     dimension. This is why the order-1 table already carries tau^2 terms and the order-2 table
     (Appendix A) carries tau^3 terms (the DDD, BDD, BBD, ... pieces).
  6. APPROXIMATE COMPRESSION (paper Sec 6, Alg. 4): optionally merge (1,2)&(2,3) etc. to shrink
     the bond further while staying accurate to order N. (Our code does NOT take this last step;
     it keeps the clean 3-level Appendix-A form, hence bond 1+chi+chi^2.)

ORDER / BOND-DIMENSION TABLE (paper Sec 8.3; chi = bond dim of the Hamiltonian's A block):
     order 1 : bond 1 + chi              <-  == WII == our makeW(Algorithm"WII")
     order 2 : bond 1 + chi + chi^2      <-  == VD2 == our makeW(Algorithm"VD2")
     order 3 : bond 1 + 3chi + chi^2 + chi^3
     order 4 : bond 1 + 5chi + 4chi^2 + chi^3 + chi^4
  "order n" means: captures every CONNECTED cluster of up to n overlapping local terms exactly,
  plus all disconnected products of such; one-step (phase) error O(tau^{n+1}).

THE NOMENCLATURE POINT (important; corrects an earlier loose "both 2nd order" label):
  - The paper states EXPLICITLY (Sec 8.2): "the first-order MPO is exactly the same as the WII
    operator from Ref. [7]" (Ref. [7] = Zaletel 2015). So WII = Van Damme ORDER 1, VD2 = ORDER 2.
    They differ by one order. WI is the cruder, pre-extension I + tau*W_H (paper Eq. 27).
  - WHY WII can look "2nd order": its one-step error is O(tau^2) generically, BUT for a STRICTLY
    nearest-neighbour H every pair of local terms overlaps on AT MOST ONE site, and WII already
    captures all single-site overlaps exactly -> for NN models WII is EFFECTIVELY 2nd order. The
    paper flags exactly this non-genericity (Sec 8.2: "the error for the second-order MPO scales
    according to a third-order MPO ... not generically true ... depends on the Hamiltonian").
  - OUR MODELS ARE NNN: local terms (the ZZ_{i,i+2} bridge, the pXX, etc.) overlap on TWO sites,
    which WII does NOT capture at tau^2 -> for the Alcaraz/benchmark model WII is GENUINELY only
    1st order, while VD2 is unconditionally 2nd order. THIS is the physical reason VD2 is our
    production scheme. (Empirically VD2==TDVP at dt=0.05; WII agrees only to its lower order.)

VERIFIED CORRESPONDENCE (code  <=>  Appendix A, second-order table):
  The paper's "{...}" denotes the SUM OVER ALL DISTINCT PERMUTATIONS of the listed operators
  (e.g. {BD}=BD+DB, {CCD}=CCD+CDC+DCC, {ABC}= all 6). This is EXACTLY our `sym_sum`. With the
  code's tau-scaling mat_D=tau*D, mat_C=tC*C=sqrt(tau)*C, mat_B=tB*B=sqrt(tau)*B, mat_A=A, each of
  the nine blocks W11..W33 matches Appendix A after a virtual-bond GAUGE G = diag(1, sqrt(tau)*I_chi,
  tau*I_{chi^2}) (i.e. W_code[i,j] = G[i]^{-1} W_paper[i,j] G[j]); the gauge is the sqrt(tau)
  time-step splitting and cancels in the contracted MPO. Spot-checks that pin it down:
     W11 = I + tau D + (tau^2/2) D^2 + (tau^3/6) D^3                         (gauge 1, exact match)
     W22 = A + (tau/2)({BC}+{AD}) + (tau^2/6)({CBD}+{ADD})                   (gauge 1, exact match)
     W33 = AA + (tau/3)({ABC}+{AAD})                                        (gauge 1, exact match)
     W12 = sqrt(tau)*[ C + (tau/2){CD} + (tau^2/6){CDD} ]                    (= sqrt(tau) x paper W12)
     W21 = (1/sqrt(tau))*[ tau B + (tau^2/2){BD} + (tau^3/6){BDD} ]          (= paper W21 / sqrt(tau))
     W13 = tau*[ CC + (tau/3){CCD} ]                                        (= tau x paper W13)
  All nine were checked; prefactors (1, 1/2, 1/6, 1/3) and permutation sums agree exactly. CONCLUSION:
  makeW(::Algorithm"VD2") is a faithful transcription of the paper's Appendix-A 2nd-order MPO, so the
  thesis may cite the VD2 propagator as "the second-order MPO of Van Damme et al. (2024), Appendix A".

---

## 6. ITRANSVERSE.jl — KEY DATA STRUCTURES

VERIFIED field names below are from ITransverse_source/src — they differ from earlier
drafts of this file (no `expH_func`, no `rot_inds`; PMParams.opt_method is a Symbol, etc.).

tMPOParams  (tmpo/tmpo_params.jl — @kwdef struct, ALL FIELDS ARE KEYWORD ARGS):
  dt      :: Number       time step size (we use 0.1)
  dbeta   :: Number       imaginary-time step; DEFAULT = dt, we override to -im*dt
  mp      :: ModelParams  model parameter struct (AlcarazParams / IsingParams / ...)
  scheme  :: ExpHRecipe   which expmpo kernel to use (e.g. AlcarazVD2(), Murg())
  nbeta   :: Int          number of imaginary-time cooling sites (MUST be even, or 0)
  bl      :: ITensor      initial state |psi_0> as a local tensor (the tMPO left boundary)
  CONSTRUCTOR (verified against installed version 0.56.3):
    tMPOParams(mp=mp, dt=dt, nbeta=0, scheme=AlcarazVD2(), dbeta=-im*dt, bl=init_state)
  IMPORTANT: the installed version uses @kwdef (ALL kwargs, including mp=). The ITransverse_source
  has a DIFFERENT constructor form (positional mp). Always use the kwarg form shown above.
  NOTE: the old keyword `init_state` does NOT exist in the installed version — use `bl` directly.

FwtMPOBlocks  (tmpo/fw_tmpo_blocks.jl — struct):
  Wl, Wc, Wr            ITensor   left, bulk, right tensors of the (rotated) U(dt) tMPO
  Wl_im, Wc_im, Wr_im   ITensor   imaginary-time counterparts (built from dbeta)
  tp     :: tMPOParams
  iL, iR :: Index   the (rotated) virtual/link indices of the bulk tensor
  iP, iPs:: Index   the (rotated) physical indices (iPs = iP')
  There is NO `rot_inds` Dict — the rotation is encoded directly in these named indices.
  Helpers: linkinds(b)=(iL,iR), siteinds(b)=(iP,iPs), get_Ws(b; imag=false).
  Construct with FwtMPOBlocks(tp::tMPOParams).  The spatial->temporal physical dimension is
  read as  dim(inds(b.Wc, "Site,time")[1]).

FoldtMPOBlocks  (tmpo/fold_tmpo_blocks.jl):  folded forward x backward blocks WW = W (x) W_dag,
  plus imaginary-time counterparts and a folded initial density tensor; used for expectation
  values of local operators (the "folded" picture), not for the bare Loschmidt echo.

PMParams  (power_method/pm_params.jl — Base.@kwdef mutable struct):
  truncp          :: NamedTuple   e.g. (; cutoff=1e-14, maxdim=256, alg="RTM")
                                  — `alg` is the STRING that selects the LR truncation routine.
  opt_method      :: Symbol       default :sym. ONLY read by powermethod_op / powermethod_sym.
                                  powermethod_lr IGNORES it (silent no-op). We pass :nosym.
  itermin         :: Int          default 20
  itermax         :: Int          default 600 (we set 5000)
  eps_converged   :: Float64      stop when Delta(singular values) < eps_converged (we use 1e-6)
  maxdims         :: range/vec     bond-dim schedule, e.g. 2:2:maxdim (gradual chi growth)
  cutoffs         :: vector        cutoff schedule, e.g. [cutoff]
  normalization   :: String       "overlap" (renormalize by overlap_noconj each step) or "norm"
  compute_fidelity:: Bool         track <R|R_prev> each step
  stuck_after     :: Int          declare :stuck after this many non-improving steps (we use 200)
  quiet           :: Bool

IsingParams  (chain_models/model_params.jl — @kwdef mutable struct):
  Jtwo     :: Float64   default 1.0   (σ_x·σ_x coupling)
  gperp    :: Float64   default 0.4   (σ_z field)
  hpar     :: Float64   default 0.0   (σ_x field / parallel)
  phys_site:: Index     default Index(2,"S=1/2")
  Critical Ising: IsingParams(1.0, 1.0, 0.0)
  NOTE: Ising convention is σ_x coupling, σ_z field — OPPOSITE to Alcaraz (σ_z coupling, σ_x field).
  Scheme dispatch: default_scheme(::IsingParams) = Murg()
  The Murg scheme (expH_ising_murg) produces a SYMMETRIC MPO (left-right and physical-virtual),
  enabling powermethod_sym and the full n→1 entropy machinery. This is the paper's Eq. (21).

ConeParams: light-cone driver params (truncp, opt_method, which_evs, which_ents, vwidth, ...).
  Used for local-operator expectation values via run_cone, not for the plain Loschmidt echo.

---

## 7. ITRANSVERSE.jl — FUNCTION API AND USAGE RULES

### MOST CRITICAL RULE: OVERLAP WITHOUT CONJUGATION

In all transverse contraction setups, the left tMPS <L| is already a bra.
Overlap must be computed WITHOUT complex conjugation:

  CORRECT:  overlap_noconj(L::MPS, R::MPS)  -> <L|R> (no conjugation)
  WRONG:    inner(L, R)                      -> <L*|R> (applies conjugation to L)

Using inner() instead of overlap_noconj() is the most common source of wrong results.
This is not obvious because inner() is the standard ITensors function for overlaps.

### Building the tMPO

For Loschmidt echo (forward evolution only), the verified signatures are:
  fw_tMPO(b::FwtMPOBlocks, time_sites::Vector{<:Index};
          bl=b.tp.bl, tr=b.tp.bl, init_beta_only=false)  -> the tMPO (an MPO over time_sites)
  fw_tMPS(b::FwtMPOBlocks, time_sites::Vector{<:Index};
          tr=b.tp.bl, LR=:right)                          -> a structured seed tMPS
  - `time_sites` are Nsteps Index objects tagged "time" whose DIMENSION = the spatial MPO's
    virtual-bond dim (= temporal physical dim; read it dynamically, see Section 5b/6).
  - `bl` (left boundary) defaults to the initial state in tp.bl; `tr` (right/trace boundary)
    we ALSO set to |X+>. Both are passed explicitly as tr=init_state in our code.
  - SEED WARNING (important, see Section 10): the structured fw_tMPS seed is symmetry-special
    and gets TRAPPED in a subdominant Z2 sector for our benchmark model. Our drivers OVERWRITE
    it with random complex tensors before calling powermethod_lr. Do not "simplify" that away.

For expectation values of local operators (folded picture):
  folded tMPO built from FoldtMPOBlocks; fold_op inserted at the folding point
  (identity for Loschmidt-type quantities). Used with the light-cone driver, not the bare echo.

### Power Method Variants — WHICH ONE TO USE

The Alcaraz MPO is ASYMMETRIC (upper-triangular FSM, reads left-to-right only).
The correct choice is:

  powermethod_lr(in_mps::MPS, in_mpo_L::MPO, in_mpo_R::MPO, pm_params::PMParams)
    -> (powermethod_both does NOT exist - this is the real function for asymmetric networks)
    -> Updates BOTH left and right tMPS independently each iteration.
    -> in_mpo_L and in_mpo_R are the tMPO columns applied to left and right boundaries.
    -> NOTE: powermethod_lr ignores PMParams.opt_method entirely (it is a silent no-op here).
       The truncation algorithm for non-symmetric networks (our case) is selected via
       PMParams.truncp.alg = "RTM" (dispatches to the dedicated LR-RTM contraction), not via opt_method.

  powermethod_sym(in_mps::MPS, in_mpo::MPO, pm_params::PMParams)
    -> Only updates one tMPS; gets the other by transposition (<L| = <R|*).
    -> Only valid if the network is LEFT-RIGHT SYMMETRIC.
    -> The Ising model with Murg MPO qualifies; Alcaraz (NNN, asymmetric FSM) does NOT.
    -> TRUNCATION ALG: truncp.alg = "RTMsym" (NOT "RTM" — that dispatch doesn't exist for sym).
       Available: "RTMsym", "naiveRTMsym", "naive", "densitymatrix".
    -> opt_method: :RTM_R or :RTM (for symmetric). This is read by the sym PM.
    -> Returns (psi_L, info_iterations). psi_L is the MPS; <L| = <psi_L|* by symmetry.

  powermethod_op(in_mps, in_mpo_1, in_mpo_O, pm_params)
    -> For expectation values in the thermodynamic limit.
    -> opt_method = "RTM_LR" for generic asymmetric case, "RTM_R" if left-right symmetric.
    -> WARNING: with opt_method="RDM", in_mpo_O is unused; only in_mpo_1 is applied.

### Convergence monitoring in PMParams
  eps_converged monitors Delta_S between consecutive tMPS iterations.
  S is computed via RDM or RTM depending on opt_method.
  Typical good values: eps_converged = 1e-6 to 1e-8.

### Light Cone

  init_cone(tp::tMPOParams, n::Int)
    -> Initializes tMPS for n time steps (no truncation yet).
    -> Use for computing expectation values of local operators.

  run_cone(psi::MPS, b::FoldtMPOBlocks, cp::ConeParams, nT_final::Int)
    -> Runs the light cone algorithm up to nT_final time steps.
    -> Applies one tMPO layer per step, then truncates.

### Temporal Entropy Functions

  gen_renyi2(psi::MPS, phi::MPS; normalization="overlap")
    -> Computes the generalized Rényi-2 entropy S_2 = -log Tr(T^2_t) at each temporal cut.
    -> psi, phi are the two power-method tMPS (psi_L, psi_R). With normalization="overlap"
       (the DEFAULT) it first divides each by sqrt(overlap_noconj(psi,phi)) — so you do NOT
       need to pre-normalize, though our drivers do anyway.
    -> Returns a vector of (generally complex) values, ONE PER INTERNAL BOND of the tMPS, i.e.
       length = Nsteps - 1 (per bond, NOT per site). Plot Re and Im separately.
    -> THIS IS THE PRIMARY ENTROPY WE WANT TO PLOT.
    -> nbeta trimming: with nbeta>0 the cooling sits on the FIRST nbeta/2 and LAST nbeta/2
       sites (split, see Section 6 nbeta note), so trim nbeta/2 bonds from EACH end to keep
       only physical real-time bonds. nbeta=0 => no trimming.

  gen_tsallis2(psi::MPS, phi::MPS; normalization="overlap")
    -> Generalized Tsallis-2 entropy (returns 1 - Tr(T^2) per bond). Same shape as gen_renyi2.
    -> Alternative to Rényi-2, less common in the literature we're following.

  SPECTRAL GAP / SUBLEADING EIGENVALUE (lambda1) BY DEFLATION (our addition; reference only — this
  single-vector deflation is SUPERSEDED by block_transfer_eigs, kept for narrative in nb5; the
  original code is in legacy/7_gap.ipynb and legacy/10_benchmark_mpo.ipynb):
    There is NO dedicated ITransverse function for the 2nd transfer-matrix eigenvalue. We get
    lambda0 from a normal powermethod_lr run (lambda0 = expval_LR(L,mpo,R)/overlap_noconj(L,R)),
    then deflate: seed a random tMPS, Gram-Schmidt it against (psi_L_0,psi_R_0) using
    overlap_noconj projections, and iterate a modified power method
    (gen_exc1_right / gen_exc1_left: apply mpo / swapprime(dag(mpo),0,1) via `applyn`, then
    subtract the component along (L0,R0) each step). lambda1 = bi-orthogonal Rayleigh quotient
    expval_LR(L1,mpo,R1)/overlap_noconj(L1,R1), with an inner()-based fallback if that overlap
    underflows. gap_ratio = |lambda1|/|lambda0| (-> 1 signals near-degeneracy / slow PM).
    NUMERICAL CARE (learned the hard way): the random deflation seed can project almost
    entirely onto (L0,R0), leaving a ~0-norm residual whose normalization produces NaNs that
    crash eigen() inside add(); and a mid-iteration step can likewise go singular. Guard with
    a retry loop on the initial residual norm AND a try/catch(ArgumentError) around the
    iteration that reports reason="numerical_error" instead of aborting the whole T-sweep.
    SUPERSEDED for production: deflation still dies AT the gap closing. Use the BLOCK POWER
    METHOD (`block_transfer_eigs`, Section 10 "RESOLVED") instead — it gets the leading k=4
    eigenvalues at once and stays well-behaved through the degeneracy. Deflation kept only as
    a reference/fallback.

  generalized_vn_entropy_symmetric(psiL::MPS; bring_gen_can=true, normalize_eigs=true)
    -> Computes the n→1 (von Neumann) generalized temporal entropy S₁ = -Σ λᵢ log(λᵢ).
    -> Only valid in the SYMMETRIC case (<L| = <R|*).
    -> This is the quantity in Carignano–Tagliacozzo Eq. (6) — the PRIMARY CFT observable.
    -> Returns a Vector{ComplexF64} of length N_bonds (one per internal bond of the tMPS).
    -> Uses diagonalize_rtm_symmetric internally; applies salpha(eigs, 1) for von Neumann.

  gensym_renyi_entropies(psiL::MPS; bring_gen_can=true, normalize_eigs=true)
    -> Returns NamedTuple (;S0, S05, S1, S2, S4) — all Rényi orders in one call.
    -> S1 = von Neumann = same as generalized_vn_entropy_symmetric.

  diagonalize_rtm_symmetric(psiL::MPS; direction=:right, bring_gen_can=true,
                             normalize_eigs=true, sort_by_largest=true, cutoff=1e-12)
    -> Diagonalizes the RTM using (complex) orthogonal matrices (Autonne-Takagi).
    -> Only valid in the SYMMETRIC case where <L| = |R*|.
    -> Returns Vector{Vector{ComplexF64}}: complex eigenvalue spectrum per bond.
    -> Use to compute arbitrary Rényi entropies or inspect the RTM spectrum.

### Built-in model expH functions (for BENCHMARKING with symmetric MPOs)

  build_expH_ising_murg(sites::Vector{<:Index}, mp::IsingParams, dt::Number)
    -> H = -sum_i [ J*sigma_x_i*sigma_x_{i+1} + g*sigma_z_i + h*sigma_x_i ]
    -> Produces SYMMETRIC MPO tensors (useful for powermethod_sym and RTM methods).
    -> IsingParams contains J, g, h coupling constants.
    -> WARNING: sigma_x coupling, sigma_z field — DIFFERENT from Alcaraz convention.

  build_expH_potts_symm_svd(sites, mp::PottsParams, dt)
  [XXZ model also available via similar function]

---

## 8. ROTATION CONVENTION (CRITICAL FOR DEBUGGING MPO CONSTRUCTION)

ITransverse performs an IMPLICIT 90-DEGREE CLOCKWISE ROTATION of the network:
- Physical indices (i_x, j_x) of the spatial MPO U(dt) -> VIRTUAL indices (alpha_t, beta_t) of tMPO
- Virtual bond indices (alpha_x, beta_x) of the spatial MPO -> PHYSICAL indices (i_t, j_t) of tMPO

After rotation:
- The temporal "sites" of the tMPS correspond to discrete time steps (each = one dt)
- The "bond dimension" direction of the original spatial MPO becomes the physical Hilbert space
- The physical Hilbert space dimension of the temporal system = bond dimension of U(dt)
  (recall: for VD2 this bond dim is 1 + chi + chi^2, larger than WI/WII — Section 5b)
- FwtMPOBlocks stores the rotated indices explicitly as the named fields iL, iR (links) and
  iP, iPs (physical) — there is no `rot_inds` dictionary.

Consequence: when feeding the MPO from ITensorExpMPOv2.jl into ITransverse, the index
structure of W tensors must be compatible with what ITransverse expects after rotation.
If there is an index mismatch here, the tMPO will be wrong but may still run silently.

---

## 9. ACCESSING THE ITRANSVERSE SOURCE CODE  (VERSION MISMATCH — read this)

CRITICAL: the notebooks RUN against the INSTALLED package, which is OLDER than the cloned
ITransverse_source/. As of this writing the installed copy is:
  ~/.julia/packages/ITransverse/8pmYI/src/        <- AUTHORITATIVE for what the code actually calls
ITransverse_source/src/ is a NEWER clone kept only as a read-only API reference; its signatures
can DIFFER (e.g. the tMPOParams constructor form). When the two disagree, the INSTALLED 8pmYI
version wins. The concrete deltas this project relies on are pinned in §6 and §7 (tMPOParams is
@kwdef/all-kwargs with `bl=`; the asymmetric power method is powermethod_lr; truncp.alg="RTM"/
"RTMsym"; FwtMPOBlocks exposes iL/iR/iP/iPs).

At the START of any debugging session involving ITransverse internals:
1. Read the relevant file in ITransverse_source/src/ to understand the expected API
2. Check the INSTALLED version in ~/.julia/packages/ITransverse/8pmYI/src/ for any discrepancy
   (the installed API is what runs — verify against it, not just the source clone)
3. Run `import Pkg; Pkg.status()` in Julia to see the installed version commit hash

To clone fresh (run once in the workspace root):
  git clone https://github.com/starsfordummies/ITransverse.jl ITransverse_source

If the package source cannot be found in ~/.julia/packages, it may also be in:
  ~/.julia/dev/ITransverse.jl/  (if added in dev mode)

---

## 10. POWER-METHOD STATUS, THE BLOCK METHOD, AND CRITICAL CODE GOTCHAS (resolved)

### The "T≈6 spike" / "DQPT" / "Z2-degeneracy" thread is CLOSED. There is no DQPT.
The earlier worry — a |Lambda0|≈2.2 spike at T≈6, suspected to be a DQPT and entangled with a
"Z2 sector degeneracy" — was a SINGLE-VECTOR power-method NON-CONVERGENCE ARTIFACT. The block
power method (below) gives a flat |Lambda0|≈0.895 with a monotonically CLOSING gap, which is the
ENTANGLEMENT BARRIER, not a phase transition. No DQPT exists for a quench TO the critical point
(marginal Fisher-zero, first DQPT time → ∞). The whole benchmark model (Alcaraz minus the XX
term) used to probe this was DROPPED; its only salvage is the MPO-exp benchmark (→ nb2) and the
block-PM introduction (→ nb5). The abandoned Z2-sector-projection idea is gone (the |X+> boundary
is purely even-parity, so the odd sector is structurally dead — sector projection cannot help).

### Still-true seeding lesson (baked into the code): use a RANDOM seed, not the structured one.
The structured `fw_tMPS` seed is symmetry-special and converges to a SUBDOMINANT eigenvalue; our
drivers (`compute_entropies`, `run_pm_diagnosed`, block PM) OVERWRITE it with random complex
tensors to reach the true dominant eigenvector. Do NOT "simplify" that away.

### compute_entropies now supports block PM via `use_block_pm=true`
`compute_entropies(mp, T; ..., use_block_pm=true, k_block=2)` uses `block_transfer_eigs` instead
of the single-vector `powermethod_lr`. The block method stays well-behaved through the gap closing
and recovers the conformal entropy dome to larger T (demonstrated in nb3 at T=5.5 where single-
vector broke). Default is `use_block_pm=false` (single-vector) for backward compatibility.
`plot_entropy_profiles` has the same `use_block_pm`/`k_block` passthrough.

### BLOCK (subspace) POWER METHOD — our standard tool for the leading spectrum
`block_transfer_eigs` (src/transverse_tools.jl) extracts the leading k=4 transfer-matrix
eigenvalues SIMULTANEOUSLY and stays well-behaved through the gap closing, where single-vector
deflation NaN-crashes (it divides by one dominant eigenvalue).

Algorithm (oblique Rayleigh-Ritz / Petrov-Galerkin subspace iteration; MPO stays asymmetric):
  - Keep k right tMPS R[1..k] and k left tMPS L[1..k] (random complex seeds).
  - Right map: applyn(mpo, R). Left map: applyn(swapprime(mpo,0,1), L) — PURE TRANSPOSE, NO dag
    (the bilinear-form adjoint, so overlap_noconj(L, mpo*R) = overlap_noconj(mpoT*L, R)).
  - Each iteration build two dense kxk pencils (non-conjugating): S[i,j]=overlap_noconj(L_i,R_j),
    M[i,j]=overlap_noconj(L_i, mpo*R_j). Solve via W = pinv(S; rtol=1e-12)*M then eigen(W)
    (NOT eigen(M,S): generalized eig emits Inf when S is near-singular). Left Ritz vectors from
    the transposed pencil, matched to theta by NEAREST complex value. Rotate the applied block by
    the Ritz coeffs to de-mix.
  - Converge on the leading n_track=2 Ritz values; cond(S) spike -> refresh a collapsed direction
    with a fresh random tMPS bi-orthogonalized via overlap_noconj projections. Returns
    (theta sorted by |.|, bi-orthonormal L, R, info).

### THREE CRITICAL IMPLEMENTATION GOTCHAS (all fixed in src/transverse_tools.jl)
  1. `mps ./ scalar` BROADCASTS over all N site tensors (scales the norm by scalar^N) -> it does
     NOT normalize. Use `normalize(mps)`, or single-tensor scalar mult `(c) * mps`. (The old
     compute_*_entropies `psi ./= sqrt(norm)` had this latent bug; it only "worked" because
     norm≈1. The new `compute_entropies` uses scalar mult.)
  2. MPS linear combos: use `+(acc, c*vec; alg="directsum")` then truncate!; the default
     density-matrix `+` runs eigen() on a near-singular matrix and NaN-crashes on degenerate sums.
  3. Generalized eig: pinv(S)*M, never eigen(M,S), for the near-singular S at the gap closing.

VERIFIED: block PM matches random-seed powermethod_lr |Lambda0| to ~1e-6; bi-orthogonality
max|<Li|Rj>_{i!=j}| ~ 1e-3 to 1e-2, max|<Li|Ri>-1| ~ 1e-15. Crash-safe sweeps cache per-T to
results/data/ (block_pm_alcaraz_p0.{0,1}.jld2).

### block_transfer_eigs efficiency kwargs (added for nb3.5; ALL backward-compatible, default=old)
  - `maxdims::AbstractVector` — OPTIONAL bond-dim RAMP: iteration `it` truncates to
    `maxdims[min(it,end)]` (cheap early iters, then hold the cap). `nothing` (default) = fixed maxdim.
  - `seedL`/`seedR::AbstractVector{MPS}` — OPTIONAL WARM-START seeds (padded with random if < k).
    `nothing` (default) = random seeds. A converged pair reused here converges in a few iters; the
    high-payoff use is cross-T ladder warm-starting (needs a pad-tMPS helper — nb3.5 future work).
  `compute_entropies` forwards `maxdims`. nb3.5 benchmarks these; the eigenvalue-only (Route 2)
  sweep can drop to ~k=2/maxdim≈128/ramp/cutoff=1e-10 once verified (converged χ ≪ 256).

### Confirmed working
- DMRG central charge: c ~ 0.5 for Alcaraz at p = 0..2 (Ising class in equilibrium).
- ITensorExpMPOv2.jl MPO construction (validated by TDVP == direct-MPO agreement).
- Symmetric Ising control reproduces Carignano–Tagliacozzo (c=1/2) — see §17 and nb6.

### Documentation drift fixed (API truths)
- `powermethod_both` -> the real function is `powermethod_lr`. Initial state is |X+>, not |Z-up>.
- LR truncation selected by truncp.alg="RTM" (opt_method is a no-op for powermethod_lr).
- FwtMPOBlocks exposes iL/iR/iP/iPs, not a `rot_inds` Dict.

---

## 11. PHYSICS OBJECTIVES (what we ultimately want to compute)

Step 1 (RESOLVED — June 2026): The |λ0|≈2.22 spike at T≈6 was a single-vector PM non-
        convergence artifact. Block PM (nb5; legacy/16,17) shows |λ0|≈0.895 flat. No DQPT exists
        for quenches TO the critical point (marginal Fisher-zero condition). The monotonically
        closing gap is the entanglement barrier (eigenvalue-spectrum signature of log temporal
        entropy growth, per Carignano–Tagliacozzo 2024).

Step 2 (DONE — nb6/nb7; legacy/18, framework VALIDATED on Ising): pipeline validated against
        Carignano–Tagliacozzo Eq. (6) using ITransverse's symmetric Ising (Murg + powermethod_sym).
        Boundary exponent CONFIRMED: x₁=0.502 (p=0), 0.497 (p=0.1) = free-BC Ising (a₁=0.788≈π/4).
        Symmetric Ising CONFIRMED: T=4,8,12 reproduce Eq.(6) — Re(S) on the c=1/2 chord (peak
        0.37→0.44→0.47), Im(S)≈π/24=0.131. c=1/2 recovered two independent ways. ALCARAZ p=0.1
        Rényi-2 (nb7) is the OPEN PROBLEM: c≈0.69 (T=4), 6.0 (T=6), p=0 calib ≈0.70 — NOT 0.5,
        the asymmetric LR Rényi-2 profile is contaminated near the gap closing.

Step 3 (BLOCKED on Step 2): Extract temporal entropies with CORRECT coefficients:
        - n→1 entropy (Eq. 6): coefficient c/6, available via generalized_vn_entropy_symmetric
          (symmetric models only: p=0 Ising)
        - Rényi-2 (gen_renyi2): coefficient c/8 (NOT c/3), works for asymmetric Alcaraz (p>0)
        - MUST use nbeta>0 (β₀=0.2 → nbeta=4 at dt=0.1) for clean conformal boundary
        Sweep p = 0.0, 0.1, 0.2, 0.5, 1.0 once Step 2 validates the pipeline.

Step 4: Compare slope of Re(S_gen) vs CFT chord log[(2T/π)sin(πt/T)] — c from slope.
        Use p=0 as calibration standard (known c=1/2); the slope RATIO p>0/p=0 is the
        convention-free headline.

Step 5: Identify the "breaking point" p* where the temporal central charge departs from
        c=1/2 (if ever). The boundary exponent already shows x₁≈0.5 for both p=0 and p=0.1
        (first evidence universality survives).

Step 6: Determine whether dual-unitarity emergence survives NNN frustration.
        The paper predicts the transfer matrix becomes unitary as T→∞; check |τ₀|→1.

---

## 12. KEY PHYSICS DEFINITIONS

Loschmidt echo:
  L(t) = <psi_0 | exp(-i H t) | psi_0>
  Complex amplitude (NOT the probability |L|^2). We track the full complex number.
  Initial state |psi_0> = |X+>^N (product of single-site |X+>); see Section 2. This is the
  paramagnet polarized along the field (sigma_x) direction, i.e. the ground state of H at
  lambda -> infinity in this convention. (NOT the z-polarized state.)

Temporal MPS (tMPS):
  After transverse contraction, <L| and |R> are MPS with N_t sites.
  Each site j of the tMPS corresponds to time t_j = j * dt.
  Bond dimension chi of the tMPS grows with time; we truncate to keep it manageable.

Temporal cut:
  A bipartition of the tMPS at site j (separating times 0..j from j..N_t).

Reduced Transition Matrix (RTM) at cut j:
  T_j = Tr_{complement} [ |R><L| ] / <L|R>
  Non-Hermitian, complex eigenvalues, unit trace.
  NOT the same as the reduced density matrix (which would use |L><L| or |R><R|).

Generalized Rényi-2 temporal entropy at cut j:
  S_2^gen(j) = -log Tr(T_j^2)
  Computed efficiently via two copies of the RTM (see gen_renyi2 function).
  Generally a complex number; plot Re and Im separately.

CFT prediction (Ising critical point, c=1/2):
  At long times after a quench to the critical point, S_2^gen grows logarithmically.
  S_2^gen(t) ~ (c/3) * log(t) + const   [exact coefficient from arXiv:2405.14706]
  Logarithmic growth = polynomial computational cost = efficient transverse contraction.

---

## 13. CONVENTIONS CHECKLIST (verify in src/models.jl before any simulation)

Hamiltonian sign:   H = -sum_i [...]   (NEGATIVE overall sign; all terms subtracted)
Critical point:     lambda = 1
Time evolution:     U(t) = exp(-i H t)   (real time, unitary operator)
Loschmidt echo:     L = <psi_0 | U(t) | psi_0>   (complex amplitude, no absolute value)
Initial state:      |X+>^N  (NOT z-polarized) — same in TDVP and transverse
tMPS overlap:       overlap_noconj(L, R)  NOT inner(L, R)
Pauli matrices:     Standard ITensors SpinHalf site convention ("Up"/"Dn" along z); the
                    field couples to sigma_x, so we work with the "X+"/"X-" states.
Trotter step:       dt = 0.1 in current block-PM / entropy runs (0.05 for some TDVP benchmarks)
Trotter order:      2nd order (VD2 / Van Damme), unless testing with 1st order (WI)
PM seed:            random complex tensors (NOT the bare structured fw_tMPS — Z2 trap)
PM truncation alg:  truncp.alg = "RTM" for asymmetric LR PM; "RTMsym" for symmetric PM
PM asymmetric:      powermethod_lr with opt_method ignored (no-op)
PM symmetric:       powermethod_sym with opt_method = :RTM_R, truncp.alg = "RTMsym"
nbeta:              USE nbeta≥4 (β₀=0.2) for CFT-quality entropies — nbeta=0 is insufficient
                    (no clean conformal boundary state). CRITICAL LESSON from nb6/nb7 (legacy/18).
Julia indexing:     1-based arrays
Bond dimension chi: typical starting range 16 to 64 for early tests, 128+ for production
                    (maxdim=256 in the benchmark); remember VD2 temporal phys-dim = 1+chi+chi^2
Convergence check:  run power method until Delta_S < 1e-6 before extracting entropies
Project env:        run Julia with --project=. (the repo's Project.toml) or scripts error
                    with "Package ITensors not found in current path"
Entropy coefficients (Carignano–Tagliacozzo Eq. 6):
  n→1 (generalized vN):  Re(S_gen) = s₀ + (c/6)·log[(2T/π)sin(πt/T)];  Im(S_gen) = πc/12
  Rényi-2 (gen_renyi2):  coefficient c/8 = (c/12)(n+1)/n at n=2. NOT c/3.
  The old c/3 coefficient in prior CLAUDE.md versions and notebooks 16-17 was WRONG.
Boundary convention:
  |X+⟩ = FREE BC (x₁=1/2, fast convergence) in the Carignano–Tagliacozzo paper
  |Z±⟩ = FIXED BC (x₁=2, slow convergence) — different physical quench, optional cross-check
  This is because of the σ_x↔σ_z convention swap: our field is σ_x, paper's is σ_z.
  Confirmed by the boundary-exponent fit (nb7 / legacy/18): x₁(p=0)=0.502, x₁(p=0.1)=0.497 ≈ 1/2.

---

## 14. CLAUDE CODE SESSION TEMPLATE

Paste this at the start of each Claude Code session:

"Read CLAUDE.md. The ITransverse.jl source is in ITransverse_source/src/ — read the
relevant files there to verify any function signature before using it.

Today's task: [ONE SPECIFIC TASK — e.g., 'fix the overlap function call in the benchmark
notebook so it uses overlap_noconj instead of inner, then rerun and compare to TDVP'].

Before writing code: check ITransverse_source/src/ for the exact signature.
After any change: run the minimal test to verify the fix, do not restructure other code."

---

## 15. REFERENCE READING ORDER FOR CLAUDE CODE

When debugging ITransverse integration, read source files in this order:
1. ITransverse_source/src/ITransverse.jl   (main module, exports list)
2. ITransverse_source/src/tMPO.jl          (tMPOParams, FwtMPOBlocks, fw_tMPO construction)
3. ITransverse_source/src/truncations.jl   (TruncParams, RDM/RTM truncation algorithms)
4. ITransverse_source/src/powermethod.jl   (PMParams, all powermethod_* variants)
5. ITransverse_source/src/entropies.jl     (gen_renyi2, gen_tsallis2, diagonalize_rtm_symmetric)
6. ITransverse_source/src/overlaps.jl      (overlap_noconj, expval_LR)

Actual filenames may differ — check the src/ directory listing first.

---

## 16. KEY REFERENCES

Carignano 2026 (ITransverse paper): "The ITransverse.jl library for transverse tensor
  network contractions", arXiv:2509.03699. Full paper is in the uploads.

Carignano & Tagliacozzo 2024: "Loschmidt echo, emerging dual unitarity and scaling of
  generalized temporal entropies after quenches to the critical point", arXiv:2405.14706.
  FULL PAPER AVAILABLE as carignano-tagliacozzo.md in the project root.
  This is the primary physics reference. KEY FORMULAS:
    Eq.(2): transfer matrix T = exp[(-κ + πL₀/(Tv))·(1 + 2iβ₀/T)]
    Eq.(3): Im(λ₀)/T = a₀ + κ/T² + a₄/T⁴  →  c = 24v|κ|/(πδt)  [eigenvalue c-extraction]
    Eq.(4): Im(λ₁-λ₀) = -πx₁/(vT)  [boundary exponent extraction]
    Eq.(6): S_gen = s₀ + iπc/12 + (c/6)·log[(2T/π)sin(πt/T)]  [n→1 temporal entropy]
  Uses β₀=0.2,0.4,0.6 (nbeta=2β₀/δt cooling steps); free BC = |↑⟩ (σ_z eigenstate in their
  convention = our |X+⟩ after σ_x↔σ_z swap), x₁=1/2. Fixed BC = |+⟩ = our |Z±⟩, x₁=2.
  Ising: v=2, c=1/2, s₀≈0.3. Potts: v≈√(3·3/2), c=4/5.

Carignano, Marimón, Tagliacozzo 2024: "On temporal entropy and the complexity of
  computing the expectation value of local operators after a quench", Phys. Rev. Research
  6(3), 033021. Foundational paper for RTM truncation and generalized temporal entropies.

Van Damme, Haegeman, McCulloch, Vanderstraeten 2024: "Efficient higher-order matrix product
  operators for time evolution", SciPost Phys. 17, 135 (2024); arXiv:2302.14181. FULL PDF is in
  the uploads (shared 2026-06; verified term-by-term against the code, see Section 5c.7). The VD2
  exponentiation kernel we use in ITensorExpMPOv2.jl (eulerbuilder.jl makeW(::Algorithm"VD2",...);
  see Sections 5b, 5c, 5c.7) IS their Appendix-A second-order MPO. Their order-1 MPO == Zaletel
  WII; bond-dim table (Sec 8.3): order n bond = 1+chi (n=1), 1+chi+chi^2 (n=2).

Zaletel, Mong, Karrasch, Moore, Pollmann 2015: "Time-evolving a matrix product state with
  long-ranged interactions", Phys. Rev. B 91, 165112. Origin of the WI/WII exponential-MPO
  constructors (the two-auxiliary-boson trick); see Section 5c.3.

Alcaraz (original): model definition and self-duality proof, finite-size scaling confirming
  Ising universality class for p <= 1.5.

---

## 17. RESOLVED CAMPAIGN HISTORY + CFT VALIDATION RESULTS (June 2026)

### THE "DQPT CAMPAIGN" IS CLOSED — there is no DQPT (historical record)

An early campaign chased |λ0|≈2.2 at T≈6, suspected to be a DQPT, and explored a Z2-sector
projection to explain a near-degeneracy. Both threads are RESOLVED and the relevant notebooks are
now in legacy/. Conclusions worth keeping:
  - **No DQPT.** The spike was a single-vector PM non-convergence artifact; block PM gives a flat
    |λ0|≈0.895 with a monotonically closing gap (gap_ratio 0.5→0.995 over T=0.5..7) for BOTH p=0
    and p=0.1. The quench is TO criticality (λ=1), not across it (marginal Fisher-zero, first DQPT
    time → ∞). The closing gap is the universal ENTANGLEMENT BARRIER, not a transition — it is
    qualitatively the same for p=0 and p=0.1 (not an NNN effect), but it **closes faster at larger
    p** (the barrier sets in earlier as NNN frustration grows — confirmed in nb5).
  - **Z2 projection cannot help** and is abandoned: the |X+⟩ boundaries are purely even parity, so
    the odd sector is structurally dead; the near-degeneracy is WITHIN the even sector. Block PM
    (not sector projection) is the correct tool.
  - TDVP rate ℓ(T) for N=40,80 is also flat (~0.10), no kink — independent confirmation.

### CONFIRMED RESULTS ACROSS THE CAMPAIGN

  1. Boundary exponent x₁ (Eq. 4): x₁(p=0)=0.502, x₁(p=0.1)=0.497, a₁=0.788≈π/4 → matches
     free-BC Ising exactly. Confirms |X+⟩ = free BC, and p=0.1 keeps Ising-class boundary content.
  2. Entropy dome shapes are correct (chord-like profile), but the first c-extraction (nb 17) was
     WRONG: used c/3 (should be c/6 for n→1, c/8 for Rényi-2), ran with nbeta=0 (no UV regulator).
  3. The symmetric Ising pipeline (IsingParams + Murg + powermethod_sym + generalized_vn_entropy_
     symmetric) reproduces the paper at T=4, β₀=0.2: Im(S)≈0.14 (target π/24=0.131), Re(S_mid)≈0.37.

### WHAT WENT WRONG IN THE EARLY ATTEMPTS

  - The |λ0|>1 values were non-converged Rayleigh quotients from single-vector PM, not physics.
  - The "DQPT" hypothesis was wrong: quench-to-criticality has no DQPTs at finite time.
  - The c/3 entropy coefficient was guessed from CLAUDE.md §12 instead of read from the paper.
  - nbeta=0 was used everywhere (no UV regulator β₀), making the conformal boundary condition dirty.
  - Eq.(3) eigenvalue c-extraction on nbeta=0, short-T data gives garbage (c≈5): the tiny universal
    κ/T² term is swamped by non-universal constants. Needs β₀>0 + longer T (paper uses T=14).

### CFT VALIDATION — RESULTS (these became notebooks 6 and 7 of the clean series)

The old notebook-18 staging (Eq.3 eigenvalue c, Eq.4 boundary exponent, symmetric Ising ground
truth, asymmetric Alcaraz entropy) is RUN. Findings:
  - Eq.(4) boundary exponent (CONFIRMED): x₁=0.502 (p=0), 0.497 (p=0.1), a₁=0.788≈π/4. |X+⟩ = free
    BC; p=0.1 keeps Ising-class boundary content. Robust even on nbeta=0 (phase diff cancels
    non-universal constants).
  - Symmetric Ising ground truth (CONFIRMED ✅ → NB6): Murg + powermethod_sym +
    generalized_vn_entropy_symmetric, β₀=0.2, T=4,8,12 reproduces Eq.(6): Re(S) on the c=1/2,
    s₀=0.3 chord (peak 0.37→0.44→0.47), Im(S)≈0.13–0.14 ≈ π/24=0.131. c=1/2 two independent ways.
    Figure: results/imgs/cft_ising_validation.png.
  - Emergent dual unitarity (CONFIRMED → NB6): the dominant eigenvalue λ₀(T) traced in the complex
    plane lies on a circle of near-constant radius with a winding phase — the fingerprint of the
    transfer matrix approaching a rescaled unitary. Figure: results/imgs/ising_lambda0_circle.png.
    Data cached in results/data/ising_lambda0.jld2.
  - Alcaraz p=0.1 Rényi-2 (→ NB7, OLD result): the first headline came out BAD — c≈0.69 (T=4),
    c≈6.0 (T=6); even the p=0 calibration gave c≈0.70. Data: results/data/cft_renyi2_beta_p5.jld2.
    DIAGNOSED in NB3: that cached run used an UNDER-CONVERGED block PM (itermax=800) that stalls at
    the gap closing → contaminated eigenvectors. NB7 was REMADE (June 2026) to recompute with the
    well-converged block PM (itermax=8000, nbeta=4) and to add the leading-eigenvalue route (Eq.3
    c, Eq.4 x1, λ0 circle), with p=0 as a convention calibration (headline = p=0.1/p=0 ratio).
    The remade NB7 is built but NOT yet run — c(p=0.1) is PENDING the master sweep
    (results/data/nb7_alcaraz_block.jld2). Update this section with the outcome once it runs.

### NB3 BLOCK-PM ENTROPY IMPROVEMENT (June 2026 revision)
  - The single-vector `powermethod_lr` stops converging once the transfer-matrix gap closes (T≳5
    for Alcaraz p=0.1); the Rényi-2 profile peels off the conformal dome.
  - `compute_entropies(...; use_block_pm=true)` uses `block_transfer_eigs` and recovers the dome
    to larger T. Demonstrated head-to-head at T=5.5: single-vector contaminated, block PM clean.
  - This finding feeds into the NB7 revision (deferred): applying the block-PM entropy route may
    improve the Alcaraz c-extraction that currently comes out badly.

STATUS: framework VALIDATED on the symmetric Ising case (NB6). The asymmetric Alcaraz temporal-c
(NB7) is the OPEN PROBLEM: the asymmetric LR Rényi-2 route does not yet yield a clean c at p>0.
NB7 is DEFERRED for revision after the NB3 block-PM findings mature.
Likely directions: block-PM entropy profiles at larger T (now available via use_block_pm=true);
the Eq.(3) eigenvalue c with β₀>0 + longer T (eigenvalues are more robust than contaminated
eigenvectors); higher maxdim/longer T to clean the profile; dt-convergence. The headline
c(p=0.1) is NOT yet established.

### DEFERRED

  Stage D (dt-convergence): extract c at dt=0.05 to bound Trotter error. Low priority until Part 4
    validates the pipeline.
  Full p-sweep (0,0.1,0.2,0.5,1.0): map c(p) and locate the breaking point p* where temporal
    universality departs from Ising (if ever). Blocked on Step 2 validation.

### KEY TOOLS AND THEIR LOCATIONS

  src/transverse_tools.jl (was dqpt_diagnostics.jl; load via include("src/thesislib.jl")):
    build_alcaraz_tmpo(T; p, lambda, dt, nbeta, MPO_alg) → (mpo, scaffold)
    block_transfer_eigs(mpo, scaffold; k, maxdim, ...) → (theta, L, R, info)
    lincomb_mps(coeffs, vecs; cutoff, maxdim) → MPS
    run_pm_diagnosed(T; ...) → NamedTuple with diagnostics (slimmed; Z2/DQPT-cosine signal dropped)
    compute_entropies(mp::ModelParams, T; scheme, nbeta, ...) → (bonds, re, im, L, R, mpo)
                       UNIFIED driver (replaces compute_alcaraz/tricritical_entropies)
    tdvp_loschmidt_amplitude(N, Ts; ...) → Dict with G, absG, rate per T (caches to results/data/)
    crashsafe_sweep(f, Ts; cachefile) → Dict
  src/models.jl (was main.jl): AlcarazParams/TricriticalParams, *_opsum, expH_*, ITransverse.expH
  Cached data (now under results/data/):
    block_pm_alcaraz_p0.0.jld2 (T=0.5..7, nbeta=0) — eigenvalues only
    block_pm_alcaraz_p0.1.jld2 (T=0.5..5.5, nbeta=0) — eigenvalues only
    cft_renyi2_beta_p5.jld2 (nbeta=4) — Alcaraz Rényi-2 c: p=0.1 gives c≈0.69 (T=4), 6.0 (T=6) [BAD]
