# CLAUDE.md — orientation for AI coding sessions

Dense, AI-facing map of `internship_ICFO/`. Humans should read [README.md](README.md)
(overview + numerical results) and [THEORY.md](THEORY.md) (the analytical derivation)
instead — this file just gets a future Claude session up to speed fast. The git root is
the parent `QuantumMaster/`; the sibling `master_thesis/` is a **different** project.

## What this project is

Mixing time of a **Tensor-Network Metropolis–Hastings** sampler for the 2D classical
Ising model. The proposal `q` is the Boltzmann distribution approximated by a
boundary-MPS contraction truncated to bond `D_bound`; it's the proposal of an
**Independence MH** chain, so mixing is governed by `C = max_x π(x)/q(x)`
(Mengersen–Tweedie; Liu: `λ₂ = 1 − 1/C` exactly). Reference paper:
`SciPostPhys_14_5_123.md` (machine-converted; search with `LC_ALL=C grep -a`).

## Current status (2026-06-20)

- **Numerical program: complete.** Worst case `t_rel = C` exact; `C ~ ρ(Ly,D,β)^{Lx}`,
  `log ρ ∝ Ly` at β_c ⇒ exponential in volume but tiny exponent (`ρ−1≈0.04` at D=2,Ly=8).
  Typical case mild (`χ² ≪ C`, `τ_int≈0.5` to L=8) ⇒ fast in practice. Validated by CFTP
  (0.9%) and a 5.8× Glauber speedup. All results in `results/*.jld2` + the headline PNGs.
- **Analytical program: per-row result derived + verified (Step A).** Exact closed form
  `r(s) = E_p[g]/g(s)`, `g(s)=a_D(s)/a(s)`; `ρ−1 = ` relative spread of `g` (confirmed to
  0.3% over 30 regimes); the mixing penalty is the *variation* of the truncation across
  configs, not its size. Tightened bound `|η|≤ε_D‖P_disc v‖‖β‖/⟨v|β⟩` holds (gains only
  1.2–2.9× over crude). `ρ−1` tracks `ε_D`; boundary entanglement → log2 in the ordered
  phase (so ρ peaks just *above* β_c, not at it). **Open:** a tight a-priori constant, a
  clean `ε_D(D)` bound, and **Step B — lift the per-row bound to the lattice**
  (`log C = Σ_x δ_x`, the accumulation curve).

## File map

```
README.md  THEORY.md  CLAUDE.md  SciPostPhys_14_5_123.md
jl/icfo_intern/
  main.jl                 Ising PEPS, boundary-MPS proposal sampler, IMH sweep
  tools/tnmh_tools.jl     analysis primitives (see below)
  analysis.ipynb          THE results notebook — recomputes every result from code, with
                          exposed params + objective/software-eng explanations (~20–30 min full)
  TNMH_test.ipynb         the sampler in action (phase-transition sweep)
  results/                8 source .jld2 + 6 headline .png
  ising_mcmc_results_D2.jld2   original L=16,32,64 sweep (used by TNMH_test)
py/                       superseded Python prototype + a separate course assignment
```

## Code primitives — `jl/icfo_intern/tools/tnmh_tools.jl`

(assumes `include("main.jl")` first)

- `enumerate_weights(Lx,Ly,β,D)` → all `2^N` configs: `logq, logpi, logw, E, C=max w,
  argmax, sum_q (≈1 check)`. The brute-force `C` oracle (keep `N ≲ 20`).
- `proposal_logprob(config,…)` / `proposal_logprob!` — deterministic `log q` of a pinned config.
- `accumulation_curve(config,Lx,Ly,β,D)` → running `log r_k` per row (`log C = Σ δ_x`).
- `perrow_weights(Lx,Ly,β,D)` → per-config exact/truncated row weights `a(s), a_D(s)`
  (⇒ `g(s)`, the spread, the per-row analytical result).
- `compute_rho_plateau` / `rho_tracker` — the per-layer rate `ρ` (saturated / vs depth).
  **Note:** these report the *unnormalized* ratio `max a/a_D`; the mixing-relevant
  conditional rate is `E_p[g]/min g` (slightly smaller — see THEORY.md).
- `truncation_error_fidelity` → `ε_D = √(1−F²)`, fidelity distance exact↔truncated env.
- `imh_kernel` / `lambda2` / `tv_decay` — IMH kernel + spectrum (Liu check).
- `weight_samples`, `chi2_*`, `restricted_C`, `integrated_autocorr`, `cftp_coalescence`,
  `glauber_tau_int` — typical-case + validation diagnostics.

## Run

```bash
cd jl/icfo_intern && julia --project=.     # first time: Pkg.instantiate()
```
Env pinned in `jl/icfo_intern/{Project,Manifest}.toml` (ITensors 0.9.25, ITensorMPS
0.3.45, JLD2, Plots, ProgressMeter). `analysis.ipynb` recomputes from code (~20–30 min
full; sections independent, slow cells marked ⏱). The `results/*.jld2` are the canonical
full-size data behind the README figures; the notebook does not overwrite them.

## Gotchas

- **`ρ` vs `C`** (historical): an earlier `simulation.ipynb` measured the per-layer rate
  `ρ` and labeled it `C`. They differ: `C ~ ρ^{Lx}`. Current code/docs name them right.
  That superseded notebook is gone (its valid part survives as `TNMH_test.ipynb`).
- **`py/` is a superseded prototype** (Julia is canonical). Same math, not maintained.
- **`[1,1]` cap** in the `ρ` trackers marginalizes the rows above the measurement row,
  giving a *marginal* (not conditional) ratio — harmless at the sizes used (verified
  marginal ≈ conditional to 0.1%), but the conditional `E_p[g]/min g` is the exact object.
