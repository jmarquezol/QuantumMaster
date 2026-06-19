# RESEARCH_PLAN.md — Answering the TNMH mixing-time question

**For Claude Code, run from `internship_ICFO/`.** Read `CLAUDE.md` first; this plan
assumes its definitions (`π`, `q`, `w=π/q`, `C=max_x w`, `ρ`, the `C~ρ^{Lx}`
relation, the ρ-vs-C conflation in the notebook). The Julia code in
`jl/icfo_intern/` is canonical; `py/` mirrors it and is useful for independent
cross-checks.

## Goal (what "answering the question" means)

Produce data + figures that substantiate this two-part quantitative answer to
"what is the mixing time of the TNMH chain?":

1. **Worst case (exact):** the relaxation time is `t_rel = C` exactly
   (Liu: `λ₂ = 1 − 1/C`), and `C ~ ρ(Ly,D,β)^{Lx}` — exponential in volume at
   criticality for any fixed `D < D_exact`.
2. **Average case (explains practice):** typical mixing is governed by
   `χ²(π‖q) = Var_q(w)` and the bulk of the weight distribution, which stay mild
   while `C` is dominated by an exponentially rare configuration.

Every phase ends with saved artifacts (`.jld2`/`.npz` + `.png`) and a one-paragraph
finding appended to `results/FINDINGS.md`. Treat the plan as **exploratory**: each
phase lists several methods ("possibilities"); try them, keep what gives clean
results at feasible sizes, and report what failed and why.

## Operating principles

- **Reproducibility:** fixed RNG seeds; every figure regenerable from a saved data
  file; write a `results/` directory (data + figs + `FINDINGS.md`).
- **Enumeration limits:** brute-force over configs is feasible to `Lx·Ly ≲ 18–20`
  (`2^20 ≈ 10^6`). Exact environments need `maxdim = 2^{Ly}`, so keep `Ly ≲ 14`
  for the exact branch. State the size ceiling you actually used.
- **Cross-check:** validate at least two (Lx,Ly,D,β) points in BOTH Julia and
  Python; they must agree to ~1e-8 on `log q(x)` for a pinned config.
- **Correctness control (run before trusting anything):** with `D ≥ 2^{⌈Ly/2⌉}`
  the proposal is exact ⇒ `C = 1.0000` and acceptance `= 1`. Any deviation means a
  bug in the pinned-config `log q` evaluator. Reproduce this before Phase 1.
- **Update CLAUDE.md** as issues are resolved (relabel ρ; mark §9 items fixed).

---

## Phase 0 — Infrastructure (everything else depends on this)

**0.1 Deterministic proposal log-probability `log_q(config)`.** The single most
important new primitive. Add a variant of `sample_config_opt` that, instead of
drawing each row, *pins* the row to a supplied configuration and accumulates the
chain-rule `Σ log p_chosen`. Reuse the exact conditional machinery in
`sample_classical_1d` (Julia) / `sample_1d_mps` (Python) — only the `rand()` draw
is replaced by the pinned spin. Return `log q(config)` for arbitrary `config`.
- Julia: `proposal_logprob(config, Lx, Ly, beta, D_bound) -> Float64`.
- Python: same on `PEPS`.

**0.2 Exact target `log_pi(config)` and `Z`.** `log π(x) = −β·measure_energy(x) −
log Z`. Get `Z` two ways and check they agree: (a) exact boundary env at
`maxdim=2^{Ly}` (Julia) / `contract_2d_exact` (Python); (b) `compute_Z_brute_force`
for tiny lattices.

**0.3 Enumeration harness.** Iterate all `2^{Lx·Ly}` configs; return arrays of
`log π`, `log q`, `w = π/q`. This is the backbone of Phases 1–2.

**Deliverable:** `tools/` with the three primitives; the correctness control
passing (`C=1` at exact `D`) reported in `FINDINGS.md`.

---

## Phase 1 — Exact spectral verification (ground the theory in-repo)

Confirm, with code, the claims currently only asserted (CLAUDE.md §9.3).

**1.1 Brute-force `C`.** `C = max_x w(x)` from the Phase-0 harness, small lattices.

**1.2 Build the IMH kernel and diagonalize.** On the `2^{Lx·Ly}` state space,
`P[x→x'] = q(x')·min(1, w(x')/w(x))` (`x'≠x`), diagonal `= 1 − Σ`. Diagonalize;
extract `λ₂`. **Acceptance criterion:** `λ₂ = 1 − 1/C` to ≥5 decimals across
several (Lx,Ly,D,β). Feasible to ~`2^{12}–2^{14}` states (dense), more if you build
`P` sparse / use `eigs` for the top eigenvalues.

**1.3 TV-decay tightness (optional but convincing).** Iterate `P^n` from a worst
start; plot `‖P^n(x₀,·) − π‖_TV` vs `n` against `(1−1/C)^n`. Shows the
Mengersen–Tweedie bound is essentially tight.

**Possibilities / fallback:** if dense diagonalization is too big, get `λ₂` from
the power method on `P` deflated against `π`, or from the empirical
autocorrelation of a long chain (Phase 3.4) — but the exact kernel is the gold
standard; prefer it at small size.

**Deliverable:** `results/phase1_spectrum.{jld2,png}` + the `λ₂` vs `1−1/C` table.

---

## Phase 2 — Worst-case scaling: `C`, `ρ`, and the `C ~ ρ^{Lx}` law

This is the core quantitative result. Several routes to `C`; do them and check they agree.

**2.1 Relabel and finish `ρ(Ly,D,β)`** (CLAUDE.md §11.1). Rename the notebook's
`worst_case_ratios`→`rho_x`, `compute_asymptotic_C`→`compute_rho_plateau`,
`C_plateaus_*`→`rho_plateaus_*`. Sweep `ρ` over `Ly ∈ {4..20}`, `D ∈ {2,3,4,6,8}`,
`β ∈ {0.3·βc … 1.3·βc}` plus `βc`. Save. Expected: `ρ → 1` as `D → 2^{⌈Ly/2⌉}`;
`ρ ≈ 1` away from `βc`; `log ρ` roughly linear in `Ly` at `βc` (entanglement
deficit).

**2.2 Brute-force `C(Lx)` and extract the slope.** For small `Ly` (so enumeration
fits), sweep `Lx` and compute `C` via Phase 0. **Plot `log C` vs `Lx`** → expect a
line; slope `= log ρ_eff`. This is the direct confirmation of `C ~ ρ^{Lx}`.

**2.3 Reconcile the two `ρ`s (resolves CLAUDE.md §9.2).** Compare the slope
`ρ_eff` from 2.2 against the marginal `ρ` from 2.1 at the same `(Ly,D,β)`. The
tracker measures a *marginal* ratio; the slope of `log C` measures the *conditional*
per-row factor. **Test whether they're equal or `ρ_eff > ρ_marginal`.** If they
differ, the conditional is the correct per-layer object and the tracker should be
switched to condition on (rather than `[1,1]`-cap) the rows above. Report which.

**2.4 `ρ` vs truncation error (the analytic-bound avenue).** For the
single-truncation / few-row geometry, record the discarded Schmidt weight
`ε_D = (Σ_{j>D} s_j²)^{1/2}/‖s‖` at the SVD step and fit `ρ − 1` vs `ε_D`. Aim for
a relation like `ρ ≤ 1 + c·ε_D` (or whatever the data supports). This connects the
mixing constant to a measurable truncation diagnostic.

**2.5 Bond dimension needed.** From 2.1, find `D*(Ly)` such that `ρ ≤ 1 + 1/Lx`
(so `C = O(1)`), and separately the `D` for `ρ = 1` exactly. Expected story:
controlling per-layer `ρ` needs `D ~ poly(Ly)` (match `log D ≳ S_exact`), but
strict size-independent `C` needs `ρ=1` ⇒ `D ~ 2^{Ly/2}`. State both.

**Possibilities / fallback:** if you can't reach `Lx` large enough to see a clean
line in 2.2 by full enumeration, get `C(Lx)` instead by (a) CFTP coalescence
(Phase 4), or (b) a max-weight search (simulated annealing on `log w(x)`) which
gives a certified *lower* bound on `C` at sizes beyond enumeration — and check the
lower bound also grows like `ρ^{Lx}`.

**Deliverable:** `results/phase2_scaling.{jld2,png}`: `ρ(Ly,D,β)` surfaces,
`log C` vs `Lx` line with slope vs `ρ`, `ρ` vs `ε_D` fit, `D*(Ly)` curve.

---

## Phase 3 — Average-case / typical mixing (explain the empirical speed)

Why is practice fast when worst-case `C` is exponential? Quantify the typical case.

**3.1 Importance-weight distribution.** Sample `x ~ q` (reuse `sample_config_opt`,
which returns `log q`), compute `log w(x) = log π(x) − log q(x)`. Since `E_q[w]=1`
fixes the scale, you can work up to the `Z` constant on large lattices. Histogram
`log w` vs `(β,D,Lx,Ly)`. Expected: bulk tightly near 0 (`w≈1`) with a thin heavy
tail reaching toward `C`. Quantify bulk width and tail.

**3.2 `χ²` divergence.** `χ²(π‖q) = E_q[w²] − 1 = Var_q(w)`. Compute **exactly** on
small lattices (Phase 0 harness) and **by sampling** on larger ones. ⚠️ The
sampled `E_q[w²]` is dominated by rare large `w`, so the estimator is high-variance
and itself a heavy-tail diagnostic — report effective sample size / the largest few
`w` seen, and flag when the estimate is unreliable (that unreliability *is* a
finding: it signals the worst-case/typical gap). Plot `χ²` vs volume next to `C`
vs volume: the key figure is **`χ²` mild (ideally poly) while `C` explodes.**

**3.3 Restricted / typical-set constant (warm-start conductance).** Define a
high-probability set `S` (e.g. configs in an energy band around `⟨E⟩`, or the top
`p`-fraction under `q`); compute `C_S = max_{x∈S} w(x)` and `q(Sᶜ)`. Show
`C_S ≪ C` and scales mildly, and that escaping `S` is rare — the structure of an
s-conductance / restricted-mixing argument. Sweep the band/fraction to show the
tradeoff.

**3.4 Empirical autocorrelation (reality check).** From real `run_mcmc_sweep`
runs, estimate the integrated autocorrelation time `τ_int` of energy (and
magnetization) via windowing/batch-means. Compare `τ_int(Lx)` to `ρ^{Lx}` (worst)
and to a `χ²`-based prediction (typical). **Which does practice track?** Tie to the
acceptance dip already in `ising_mcmc_results_D2.jld2`.

**Deliverable:** `results/phase3_typical.{jld2,png}`: weight histograms, `χ²` vs
`C` vs volume, `C_S` vs band, `τ_int` vs `Lx` with both predictions overlaid.

---

## Phase 4 — Independent validations (optional, do if time allows)

**4.1 CFTP (Corcoran–Tweedie IMH perfect sampling).** Coupled chains coalesce the
first time the max-weight config `s★` (the one realizing `C`) is proposed
(prob `q(s★)=π(s★)/C` per step). Implement on `Ly=2,3` strips; measure coalescence
time; check it tracks `C/π(s★)`. Independent confirmation of `C` and a direct link
to the supervisor's "Avenue B".

**4.2 Speedup vs local dynamics.** Implement single-spin-flip Glauber/Metropolis;
measure its `τ_int` at the same `(L,β)`; quantify the TNMH speedup and compare to
the paper's reported factors. Context for why collective updates matter.

**Deliverable:** `results/phase4_validation.{jld2,png}` if attempted.

---

## Phase 5 — Synthesis

Assemble the answer. `results/FINDINGS.md` should end with the master figures and
the two-part statement, plus the bond-dimension verdict (`D*(Ly)` for controlled
per-layer rate vs exponential `D` for size-independent `C`) and the
worst-vs-typical reconciliation. Update CLAUDE.md §11 to reflect what's now done.

**Master figures to produce:**
- F1: `λ₂` vs `1−1/C` (diagonal) — `C` is the relaxation time. [Phase 1]
- F2: `log C` vs `Lx` (line) and `log ρ` vs `Ly` (line at `βc`) — the `ρ^{Lx}` law. [Phase 2]
- F3: `ρ` vs `D` (→1) and `ρ` vs `β` (peak at `βc`). [Phase 2]
- F4: weight histogram (bulk vs tail) + `χ²` vs `C` vs volume. [Phase 3]
- F5: `τ_int` vs `Lx` with `ρ^{Lx}` and `χ²`-based curves. [Phase 3]

## Suggested order & checkpoints

Phase 0 → 1 (must pass the `λ₂=1−1/C` check before proceeding) → 2 → 3 → (4) → 5.
After Phase 1, pause and report the spectrum check. After Phase 2, pause and report
whether `ρ_eff = ρ_marginal` (the §9.2 question). Keep `FINDINGS.md` updated
continuously so the narrative is recoverable if interrupted.
