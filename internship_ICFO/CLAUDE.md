# CLAUDE.md — TNMH Mixing-Time Project (ICFO internship)

Orientation file for a future Claude Code session. Dense and precise on purpose.
Read this instead of re-reading the whole repo. Where it says **[FLAG]** the code
does something subtle, wrong, or mislabeled — believe the annotation over the
variable/function name.

This file documents the directory `internship_ICFO/` (the project root, even
though the git root is the parent `QuantumMaster/`). The sibling `master_thesis/`
is a **different project** (TDVP / Loschmidt / DQPT) with its own CLAUDE.md — do
not conflate them.

> **CURRENT WORK (2026-06, in progress).** Executing [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md):
> a 6-phase pipeline answering the mixing-time question (worst-case `C ~ ρ^{Lx}`;
> average-case `χ²`). New code = `jl/icfo_intern/tools/tnmh_tools.jl` + the
> `phase0..5` notebooks + a `proposal_logprob` addition to `py/peps.py`. **Status:
> primitives written and unit-tested via CLI; the notebooks have NOT been run
> end-to-end and NOTHING is committed yet — no `results/*.jld2` data exists beyond
> the `FINDINGS.md` scaffold.** Full detail in **§12**. The headline result already
> verified at small size: `λ₂ = 1 − 1/C` to ~6e-15 (Liu).

---

## 1. Project overview

**Physical system.** Classical 2D Ising model on an `Lx × Ly` square lattice,
open boundary conditions, `H(ω) = −J Σ_⟨ij⟩ σ_i σ_j` (ferromagnetic, `J=1`),
target = Boltzmann distribution `π(ω) = e^{−βH(ω)} / Z(β)`.

**Algorithm (TNMH).** Tensor-Network Metropolis–Hastings. The Ising partition
function is written as a `D=2` PEPS (one tensor per spin, the physical leg = spin
value; summing the physical legs gives `Z`). An *approximate* sampler `q` for `π`
is built by contracting that PEPS row-by-row via a **boundary MPS** truncated to
bond dimension `D_bound`, and sampling spins site-by-site from the resulting
conditional marginals (chain rule). `q` is then used as the **proposal** of a
Metropolis–Hastings chain that replaces the **entire** spin configuration in one
move (a global / collective update), instead of single-spin flips.

**Open research question.** How does the *mixing time* of this chain scale with
`D_bound`, system size, and temperature (especially near criticality)? Why are
empirical acceptance rates high in practice even though the worst-case bound is
exponential in volume at criticality?

**Based on:** Frías-Pérez, Mariën, Pérez-García, Bañuls, Iblisdir, *Collective
Monte Carlo updates through tensor network renormalization*, SciPost Phys. 14,
123 (2023). Local copy: [SciPostPhys_14_5_123.md](SciPostPhys_14_5_123.md)
(machine-converted from PDF — words are often run together / tables mangled; use
`LC_ALL=C grep -a` to search it). Supervisor: Miguel Frías-Pérez (first author).

---

## 2. Theory cheat-sheet (resume the project from this section alone)

Let `Ω = {−1,+1}^{Lx·Ly}` be the configuration space.

- `π(x) = e^{−βH(x)} / Z` — exact Boltzmann target.
- `q(x) = \tilde π_D(x)` — the TN proposal: probability that the boundary-MPS
  chain-rule sampler outputs configuration `x`. It **factorizes sequentially**
  over rows (row `x` is sampled conditioned on rows already fixed + the
  truncated environment of rows not yet visited). In code, `log q(x)` is the
  `log_prob` returned by the samplers.
- `w(x) = π(x) / q(x)` — the importance weight. The MH acceptance for proposing
  `x'` from current `x` is `min(1, w(x')/w(x))` (state-independent proposal ⇒
  the `g`-ratio is `q(x)/q(x')`). See `run_mcmc_sweep` / the driver scripts:
  `log_acc = (log q(x) − log q(x')) + (−β(E(x') − E(x)))`.

**This sampler is an Independence Metropolis–Hastings (IMH) chain** — the
proposal `q` does not depend on the current state. For IMH, convergence is
governed entirely by the scalar **Mengersen–Tweedie constant**

>  `C = max_{x∈Ω}  π(x) / q(x)`   (the worst-case importance weight over the
>  whole configuration space).

Key facts (must stay unmissable):

- **Mengersen & Tweedie (1996):** the IMH chain is *uniformly ergodic* iff
  `C < ∞`, with `‖P^n − π‖_TV ≤ (1 − 1/C)^n`.
- **Liu (1996):** the second-largest eigenvalue of the transition kernel is
  *exactly* `λ₂ = 1 − 1/C`. So `C` is literally the relaxation time, not just a
  bound. Spectral gap `γ = 1/C`.
- Hence `t_mix ≈ C · log(1/ε)`. CFTP couples in one step with probability `≥ 1/C`.

### 2a. ρ vs C — the central distinction (read twice)

There are **two different quantities** floating around the code, and the
notebook conflates them:

- **`ρ(Ly, D, β)` — per-layer / per-row fixed-point ratio.** What the existing
  "C" / "worst-case ratio" / "Mengersen–Tweedie constant" trackers actually
  compute. It is `max` over the `2^{Ly}` spin configurations of a **single row**
  of the ratio of the exact vs truncated **single-row marginal** probability,
  evaluated once the boundary environment has reached its fixed point. It is a
  property of one layer of the strip, not of the whole lattice. Its "plateau"
  in the row index is just the environment saturating.
- **`C` — true full-lattice constant** = `max` over all `2^{Lx·Ly}` *full*
  configurations of `π(x)/q(x)`. This is the one in Mengersen–Tweedie/Liu.

Because `q` factorizes sequentially over the `Lx` rows, the worst-case weight
multiplies across rows:

>  `C ~ ρ^{Lx}`  ⇒  `log C ~ Lx · log ρ(Ly, D, β)`.

Independently confirmed during the project (status: claimed, see Next steps to
re-verify in-repo): `log C` grows **linearly in `Lx`** in a brute-force check,
and `λ₂ = 1 − 1/C` matches exact kernel diagonalization to ~5 decimals. And
`log ρ(Ly,D,β)` itself grows ~linearly in `Ly` at criticality (entanglement
deficit: `S_exact ∝ (c/3)log Ly` vs MPS capacity `log D_bound`). So the true
`log C ~ Lx·(α·Ly)` — exponential in **area/volume** at `T_c`, whereas the
notebook plots `ρ(Ly)` (exponential in **width only**) and labels it `C`.

**Bottom line for revisions:** wherever the code/notebook says it measures `C`,
`τ_corr ≤ C`, or "Mengersen–Tweedie constant", it is measuring **`ρ`**. The
qualitative story ("blows up exponentially at criticality") survives; the
identification of the plotted number with the mixing-time constant does not.

---

## 3. Repository map

```
internship_ICFO/
├── SciPostPhys_14_5_123.md     Reference paper (PDF→md, garbled tables). The theory source.
├── CLAUDE.md                   This file.
├── RESEARCH_PLAN.md            The 6-phase plan now being executed (worst-case C~ρ^Lx; average-case χ²). Drives §12.
├── jl/icfo_intern/             Julia implementation (ITensors). The ACTIVE / canonical codebase.
│   ├── main.jl                 Core: Ising PEPS, boundary-MPS proposal sampler, IMH sweep.
│   ├── simulation.ipynb        THE (historical) research notebook: phase-transition sweep + ρ/"C" trackers + volume scaling. 23 cells. Left untouched.
│   ├── ising_mcmc_results_D2.jld2   Saved phase-transition sweep results (D_bound=2). See §7.
│   ├── tools/tnmh_tools.jl     NEW (§12). Reusable primitives: deterministic log q(config), enumerate_weights→true C, IMH kernel/λ₂, clean ρ, χ²/CFTP/Glauber. Unit-tested via CLI.
│   ├── phase0_infrastructure.ipynb … phase5_synthesis.ipynb   NEW (§12). One notebook per RESEARCH_PLAN phase. WRITTEN, NOT YET RUN.
│   ├── results/                NEW. Output dir for the phase notebooks. Currently only FINDINGS.md (scaffold) — no data yet.
│   ├── Project.toml            Deps: ITensors 0.9.25, ITensorMPS 0.3.45, JLD2, Plots, ProgressMeter.
│   └── Manifest.toml           Pinned full dependency graph (1771 lines).
├── py/                         Python implementation (numpy/opt_einsum). Earlier prototype; mostly superseded by jl/.
│   ├── phase0b_python_crosscheck.ipynb   NEW (§12). Independent Python log q (uses PEPS.proposal_logprob); writes results/crosscheck_py.csv. NOT YET RUN.
│   ├── matrix_product_states.py  MPS class (QR/SVD, apply_mpo, compress, state factories).
│   ├── peps.py                 PEPS class: Ising PEPS, boundary-MPS contraction, both samplers, brute-force Z, exact contraction. (+ NEW additive proposal_logprob/sample_1d_mps_pinned, §12.)
│   ├── TNMH.py                 Driver: IMH chain using the NAIVE O(N²) sampler. 4×4, β=0.44.
│   ├── TNMH_v2.py              Driver: IMH chain using the OPTIMIZED O(N) sampler. 4×4, β=0.44. [canonical of the two]
│   ├── critial_point_sim.py    Phase-transition sweep (Python twin of simulation.ipynb cells 1–2). [sic: "critial"]
│   ├── indep_sampling.py       Pure independent sampling (no accept/reject) — sanity-checks q.
│   ├── MPS_simulation.py       Unit tests for the MPS class (GHZ/W/Ising-state norms, compression).
│   ├── PEPS_simulation.py      Benchmark exact vs approx contraction of random PEPS vs size. [HAS A BUG, §5]
│   ├── Z_simulation.py         Benchmark Ising Z approx-vs-exact vs β; brute-force vs PEPS Z. [minor bug, §5]
│   ├── TN_TimeEvol/TN_assignment.ipynb   SEPARATE course assignment (TFIM real-time TEBD + transverse contraction). Reuses the MPS class. NOT part of the TNMH research. 23 cells.
│   ├── imgs/                   critical_temp_acc_rate.png, MHTN1.png, MHTN2.png (figures).
│   ├── icfo_env/               A Windows venv (D:\… paths, py3.13.9). DOES NOT RUN ON LINUX — recreate. §10.
│   └── __pycache__/            Stale .pyc (cpython-313). Ignore.
```

**Duplication / canonical choices**

- The boundary-MPS proposal sampler exists **twice**: Julia (`main.jl`) and
  Python (`peps.py`). The **Julia version is the active/canonical** one — all
  the research experiments (ρ trackers, volume scaling) live in `simulation.ipynb`
  and call `main.jl`. The Python version is the earlier prototype.
- Within Python there are **two samplers**: `PEPS.sample_configuration` (naive
  `O(N²)`, recomputes a full contraction per site) and `PEPS.sample_config_opt`
  (`O(N)`, precomputes top/bottom environments). `sample_config_opt` is the
  canonical one; `sample_configuration` is kept only as a reference/oracle.
  `TNMH.py` (naive) vs `TNMH_v2.py` (opt) is the same split — prefer `TNMH_v2`.
- `critial_point_sim.py` (Python) and `simulation.ipynb` cells 1–2 (Julia) are
  the same experiment; the Julia one is what was actually run (results saved).

---

## 4. Function reference — Julia (`jl/icfo_intern/main.jl`)

Index convention throughout: **`x` = row index (1..Lx, vertical), `y` = column
index (1..Ly, horizontal).** The boundary MPS lives along a row (`Ly` sites); the
sweep direction is over rows (`Lx`). So `Ly` = **strip width**, `Lx` = **strip
length**. Spins are stored as `0`/`1`; mapped to `+1`/`−1` via `1 − 2·s` (so
`0 → +1`, `1 → −1`).

### `create_ising_peps(Lx, Ly, beta, J=1.0) → (A, s, v)`  (main.jl:7)
Builds the Ising PEPS. `Q = [[e^{βJ}, e^{−βJ}],[e^{−βJ}, e^{βJ}]]`,
`M = √Q` (eigendecomp) splits each bond's Boltzmann weight symmetrically between
the two sites it connects. `A :: Matrix{ITensor}` (Lx×Ly). For each site, the
tensor is the outer product of `M[spin,:]` over whichever of the 4 bonds exist
(edges use a dummy `[1.0]`, i.e. open boundaries, so edge bonds have dim 1).
- `s[x,y]` physical site indices (dim 2). `h[x,y]` horizontal bonds within a row
  (dim 2; built internally). `v[x,y]` vertical bonds between rows (dim 2).
- Returns only `s` and `v` (vertical indices tracked for the sweep); `h` is
  internal.

### `compute_bottom_envs(A, s, v, Lx, Ly, D_bound) → Vector{Union{MPS,Nothing}}` (main.jl:53)
Bottom-up boundary MPS cache. `bottom_envs[x]` = the boundary MPS representing
rows `x..Lx` traced over their physical spins and **compressed to `D_bound`**
(cutoff `1e-10`), with its open upper vertical legs renamed to `v[x-1,:]` so the
next row can attach. `bottom_envs[Lx]` = bare bottom row (no compression).
`normalize!` after each MPO apply (overall scale irrelevant to conditional
probabilities). `bottom_envs[1]` is left `undef` and never read.
- Returns: a per-row environment used to "close off everything below row `x`".

### `sample_classical_1d(row_mps, s_inds) → (sampled_spins::Vector{Int}, log_prob_row)` (main.jl:95)
1D classical sampler for one effective row MPS. Precomputes right environments
(right→left), then samples left→right: at each site `p0 = w0/(w0+w1)` with
`w0,w1 = ` contracted weights of spin-up/down; draws the spin; accumulates
`log_prob_row += log(p_chosen)`. Environment normalizations are for stability and
**cancel** in `p0`.
- Returns: the row's spins (0/1) and `log q(row | environment)` (a sum of log
  conditional probabilities). **This is a genuine conditional log-prob** — the
  upper/lower environments are real, not a `[1,1]` cap.

### `sample_config_opt(Lx, Ly, beta, D_bound, J=1.0) → (config::Matrix{Int}, log_prob_tot)` (main.jl:147)
The full 2D **proposal sampler `q`**. Precomputes `bottom_envs`; sweeps rows
top→bottom; for each row sandwiches it between the running `top_env` and
`bottom_envs[x+1]`, samples it via `sample_classical_1d`, fixes those spins into
an MPO and folds it into `top_env` (apply + `noprime` + compress to `D_bound`).
- Returns: `config` (Lx×Ly, 0/1) and `log_prob_tot = log q(config)` (sum of all
  rows' conditional log-probs). **This is the canonical `log q` used everywhere.**

### `measure_energy(config, J=1.0) → Float64` (main.jl:219)
Total Ising energy `H = −J(Σ horizontal s_i s_j + Σ vertical s_i s_j)`. Maps
`0→+1, 1→−1`. **Returns total energy, NOT per-site** (callers divide by `Lx·Ly`).

### `run_mcmc_sweep(Lx, Ly, beta, D_bound, N_samples) → (acc_rate, avg_e)` (main.jl:234)
The IMH chain. Init via `sample_config_opt`; each step proposes a fresh
configuration, accepts with `log(rand()) < (curr_log_prob − new_log_prob) − β(E'−E)`.
20% burn-in discarded. `J` hardcoded `1.0`.
- Returns: acceptance rate, and mean energy **per site** over the kept history.

### Notebook-only functions (defined inside `simulation.ipynb`, not `main.jl`)

> These are the ρ/"C" measurement routines. They `include("main.jl")` and reuse
> `create_ising_peps`. **This is where the ρ-vs-C conflation lives.**

#### `run_error_tracker(Lx, beta, D_bound; Ly=8) → (x_values, fidelities, worst_case_ratios)` (cell 9)
Sweeps rows bottom→top maintaining **two** environments in parallel: `E_exact`
(maxdim `2^{Ly}`, lossless, cutoff `1e-16`) and `E_trunc` (maxdim `D_bound`,
cutoff `1e-10`). `log_Z_exact`/`log_Z_trunc` accumulate the norms pulled out by
`orthogonalize!(·,1)` (log-Z bookkeeping so absolute probabilities are
comparable). At each row it records:
- `fidelities` = `|⟨E_exact|E_trunc⟩|` (additive error).
- `worst_case_ratios[x]` = `max` over **all `2^{Ly}` spin configs of one row** of
  `exp(log_prob_exact − log_prob_trunc)` where each `log_prob` is the **single-row
  marginal** probability under the respective environment.
- **[FLAG ρ-not-C]** `worst_case_ratios` is **`ρ_x`, a per-row marginal ratio**,
  not the full-lattice `C`. Its plateau in `x` is the environment fixed point.
- **[FLAG marginal-vs-conditional]** the measurement row's *upper* vertical bond
  is closed with `ITensor([1.0,1.0], v_inds[x-2,y])` (main.jl-style `[1,1]` cap)
  when `x-1>1`. That traces/marginalizes everything above the row, so the
  measured ratio is a **marginal** single-row ratio, not the **conditional**
  ratio `π(x)/q(x)` that actually enters `w(x)`. Whether the cap is intentional
  is an open question (§9) — for a true `C` you want the conditional, i.e. the
  product of per-site conditionals over the *whole* configuration.
- `x_values` = "rows from bottom" (`Lx − x + 1`).

#### `compute_asymptotic_C(Lx, Ly, beta, D_bound) → max_ratio::Float64` (cell 19)
Same machinery but **Phase A** just sweeps to saturate `E_exact`/`E_trunc` (no
inner loop), then **Phase B** does the `2^{Ly}` brute-force max **once** at the
true top row (`x=2`, measuring row 1 — which genuinely has no upper bond, so no
`[1,1]` cap needed there). Returns the plateau value.
- **[FLAG ρ-not-C]** the returned `max_ratio` (stored as `C_plateaus_D2/3/4`,
  labeled "Plateau Constant C") **is `ρ(Ly, D, β)`**, the per-layer rate — *not*
  the Mengersen–Tweedie `C`. Relabel on sight.

---

## 5. Function reference — Python (`py/`)

### `matrix_product_states.py` — `class MPS` (tensor convention `(d_phys, L, R)`)
- `__init__(N, d_phys, A)` — stores tensors and **immediately left-canonicalizes**
  (note: the copy in `TN_assignment.ipynb` gates this on `N==50`; the standalone
  module always does it).
- `left_canonical_form()` — QR sweep; orthogonality center → site `N−1`.
- `compute_amplitude(idx)` — coefficient ⟨idx|ψ⟩ by slicing + matrix product.
- `norm_canonical()` / `norm_general()` — norm from the center tensor / by full
  contraction.
- `normalize_tensors()` — divide each tensor by its max |entry| (stability, not a
  true normalization).
- `expectation_value(op, site_idx)` — single-site ⟨ψ|op|ψ⟩ via zipper.
- `apply_mpo(mpo)` — `|ψ'⟩ = O|ψ⟩`; MPO tensors `(d_out, d_in, Lw, Rw)`; fuses
  bonds so `χ → χ·χ_mpo`. Returns a NEW MPS.
- `compress(max_bond_dim)` — SVD sweep right→left, truncate to `max_bond_dim`,
  push `U·S` left; center → site 0.
- Factories: `create_ghz`, `create_w_state`, `create_ising_state(N, beta)` (1D
  classical Ising Boltzmann state, bond carries previous spin), `create_pauli_x_mpo`.

### `peps.py` — `class PEPS` (tensor convention `(physical, left, up, right, down)`)
Here `x` = row (outer list index), `y` = column. "Up" leg = bond to row `x−1`,
"Down" = bond to row `x+1`. Edge legs have dim 1 (open BC).
- `compute_norm(D_bound)` — `⟨Ψ|Ψ⟩` for a *quantum* PEPS (doubles the layer, MPS
  phys dim `D²`). Not used by the Ising/classical pipeline.
- `contract_2d(D_bound, fixed_config=None)` — **classical** boundary-MPS
  contraction (phys dim `D`, not `D²`). `fixed_config` is an `Lx×Ly` array with
  `−1` = sum the physical leg, `0/1` = pin that spin. This is the workhorse for
  marginals/weights.
- `contract_2d_exact()` — exact `opt_einsum` contraction (exponential; small grids
  only). Oracle for benchmarks.
- `create_random_2d_peps(...)`, `create_ising_2d(Lx, Ly, beta, ...)` — Ising PEPS,
  `W = [[e^{βJ},e^{−βJ}],[e^{−βJ},e^{βJ}]]`, `M = √W` (`eigh`); identical math to
  the Julia `create_ising_peps` (forces `d_phys=D=2`).
- `compute_Z_brute_force(Lx, Ly, beta, J)` — `O(2^{N})` exact `Z`. Oracle.
- `sample_configuration(D_bound)` — **naive `O(N²)`** proposal sampler: for each
  site, `contract_2d` with the partial config pinned for spin 0 and spin 1 →
  conditional `p0`. Returns `(config 0/1, log q)`.
- Optimized pipeline (the `O(N)` proposal, Python twin of Julia
  `sample_config_opt`): `row_to_mps`, `row_to_mpo`, `compute_bottom_env(D_bound)`,
  `eff_row_mps(x, top_env, bottom_env)`, `sample_1d_mps(row_mps)`,
  `row_to_fixed_mpo(x, spins)`, and the entry point
  `sample_config_opt(D_bound) → (config, log q)`.

**Python↔Julia correspondence**

| Concept                  | Python (`peps.py`)            | Julia (`main.jl`)        |
|--------------------------|-------------------------------|--------------------------|
| Ising PEPS               | `create_ising_2d`             | `create_ising_peps`      |
| bottom env cache         | `compute_bottom_env`          | `compute_bottom_envs`    |
| effective row MPS        | `eff_row_mps`                 | (inline in `sample_config_opt`) |
| 1D row sampler           | `sample_1d_mps`               | `sample_classical_1d`    |
| full proposal sampler    | `sample_config_opt`           | `sample_config_opt`      |
| energy                   | `measure_energy` (drivers)    | `measure_energy`         |

These are consistent algorithmically. Minor inconsistencies:
- Python `measure_energy` is defined per-driver; `indep_sampling.py`'s version
  returns energy **per site**, the others return **total**. Julia returns total.
- Spin↔index mapping is consistent (`0→+1, 1→−1`) everywhere.

### Driver scripts
- `TNMH.py` — IMH using the **naive** sampler. `Lx=Ly=4, β=0.44, D_bound=8,
  N_samples=15000`. **[FLAG]** the *initial* state is drawn with `D_bound=10`
  while the loop uses `8` (harmless but inconsistent).
- `TNMH_v2.py` — same but **optimized** sampler (`sample_config_opt`). Prefer this.
- `critial_point_sim.py` — phase-transition sweep; `run_mcmc_sweep` returns
  `(acc, avg_e/site)`; `L∈{16,32,64}`, 8 β's in `[0.3,0.6]`, `D_bound=2`,
  `N_samples=2000`. Plots acceptance & energy vs β; marks `β_c=ln(1+√2)/2`.
- `indep_sampling.py` — draws every sample fresh from `q` (no accept/reject); used
  to inspect the bare proposal. `4×4, β=0.4, D_bound=8, N=1000`.
- `MPS_simulation.py` — unit tests of the MPS class against analytic GHZ/W/Ising
  norms + a compression round-trip. `N=10`.
- `PEPS_simulation.py` — error/time of approx vs exact contraction of **random**
  PEPS vs size. **[BUG]** line 11 prints with `Lx,Ly` and line 14 builds with
  `Lx,Ly`, but only `L=11` is defined → `NameError` before the loop. Fix: use `L`
  or define `Lx=Ly=L`.
- `Z_simulation.py` — approx-vs-exact Ising `Z` vs β (`L=13`), then brute-force vs
  exact/approx `Z` (`L_small=4`). **[BUG]** line 72 appends `beta = 0.3` onto a
  comment line (dead/typo); `beta` stays `0.3` from line 54 anyway.

---

## 6. Notebook / experiment log

### `jl/icfo_intern/simulation.ipynb` — THE research notebook (23 cells)

- **Cell 0** — `using …; include("main.jl")`.
- **Cells 1–3** — Phase-transition sweep. `L∈{16,32,64}`, 8 β's in `[0.3,0.6]`,
  `D_bound=2`, `N_samples=2000`. Runtimes recorded in output: 16×16 ≈ 35 min,
  32×32 ≈ 2 h 29 min, 64×64 ≈ **10 h 58 min**. Saved to `ising_mcmc_results_D2.jld2`.
- **Cell 4** — `pwd()` shows `…/.local/share/Trash/files/jl/icfo_intern` (the
  notebook was at some point run from a copy in Trash — cosmetic, ignore).
- **Cells 5–6** — reload the `.jld2` and plot acceptance rate & ⟨E⟩/N vs β with
  the `β_c≈0.441` line. **Conclusion (solid):** acceptance rate dips around `β_c`
  and the dip deepens with `L` (e.g. `L=64` drops to ~0.18 near `β≈0.457`), but
  stays usable — consistent with the paper.
- **Cell 7** — horizontal rule (markdown).
- **Cell 8 (markdown)** — "Open problem: Mixing Times in Collective Updates."
  States the IMH classification, Mengersen–Tweedie uniform ergodicity, `γ ≥ 1/C`,
  `τ_corr ≤ C`, CFTP `≥ 1/C`. Describes the `Ly=8` strip method: exact env at
  `D=2^{Ly}=16` vs truncated at `D_bound`, measuring fidelity and a "worst-case
  ratio `C_x = max_i ⟨ψ_i|E_exact⟩/⟨ψ_i|E_trunc⟩`" over `2^8=256` boundary spin
  configs. **[NEEDS REVISION]** `C_x` here is **`ρ_x` (per-row marginal ratio)**,
  and `i` ranges over one row's `256` configs, not over `Ω`. The Mengersen–Tweedie
  statements are correct in general but are about the full-lattice `C`, not `C_x`.
- **Cell 9** — defines `run_error_tracker` (see §4).
- **Cells 10–12** — run it at `Lx=40, Ly=8`, high-T (`T=3.5`) and critical
  (`T=2/ln(1+√2)≈2.269`), for `D_bound = 2, 3, 4`. Plots ρ_x ("worst-case ratio")
  and fidelity vs rows-from-bottom. **Conclusion (partially valid):** ρ_x
  **plateaus** ⇒ the *per-layer* rate is finite for `Ly=8`. The notebook reads
  this as "computational proof of uniform ergodicity / `τ_corr ≤ C` bounded"
  — **that inference is wrong as stated**: a finite ρ does not bound the
  full-lattice `C` (which is `~ρ^{Lx}`).
- **Cell 13 (markdown)** — wider strips `Ly=10,12`.
- **Cells 14–17** — `Lx=40`, critical T, sweeping `Ly=10,12,16` (and `D_bound=12`
  at `Ly=12`). Shows the plateau **value rises with `Ly`** and **falls with
  `D_bound`**. **Conclusion (solid, modulo the ρ/C label):** entanglement-deficit
  story — `S_exact∝(c/3)log Ly` vs `S_capacity=log D_bound`.
- **Cell 18 (markdown)** — interprets the plateau as `C`, ties rising plateau to
  the entanglement deficit, predicts `log C ∝ Ly` (straight line) ⇒ "exponential
  volume scaling `C(Ly)∝exp(αLy)`." **[NEEDS REVISION]** it is `log ρ ∝ Ly`;
  ρ is exponential in **width**, and the true `C~ρ^{Lx}` adds the `Lx` factor.
- **Cell 19** — defines `compute_asymptotic_C` (returns the plateau ρ; see §4).
- **Cells 20–21** — volume-scaling experiment: `Lx_fixed=50`, critical
  `T=2.269`, `D_bound∈{2,3,4}`, sweeping `Ly` (cell 20: 4..20; cell 21: 4..32).
  Log-scale plot of plateau vs `Ly` → straight lines = exponential in `Ly`.
  **[NEEDS REVISION]** axis says "Plateau Constant C"; it is `ρ(Ly,D,β)`.
- **Cell 22 (markdown)** — "finite-size sanctuary": criticality's infinite ξ is
  capped by `ξ_D ∝ D^κ`; bounded while `Ly < ξ_D`, exponential once `Ly→∞`.
  **Conclusion (solid).**

### `py/TN_TimeEvol/TN_assignment.ipynb` — SEPARATE assignment (not TNMH research)
Real-time evolution of the 1D TFIM `H=Σ X_iX_{i+1} + gΣ Z_i`, `g=0.7`, quench
from `|↑…↑⟩`, measuring `⟨Z_{N/2}(t)⟩`, `N=50`, `χ=200`. Part I.1 brute-force
Trotter-MPO; Part I.2 TEBD (local 2-site gates, gauge center tracking); Part II
**transverse contraction** (rotate spacetime TN, spatial transfer matrix as a
power method, boundary independence via Lieb–Robinson light cone); Part III
temporal entanglement profiles + sub-linear `S_max(T)` scaling. Reuses the same
`MPS` class. Mentioned here only so it is not mistaken for TNMH work.

---

## 7. Results inventory

### `jl/icfo_intern/ising_mcmc_results_D2.jld2`  (from `simulation.ipynb` cells 1–3)
JLD2/HDF5. Keys & exact contents (verified by reading the file):
- `L_values :: Vector{Int}` = `[16, 32, 64]`
- `beta_values :: Vector{Float64}` = `[0.3, 0.4, 0.41, 0.4333, 0.4567, 0.48, 0.5, 0.6]`
- `D_bound = 2`, `N_samples = 2000`
- `results_acc :: Dict{Int,Vector{Float64}}` (per `L`, length 8 = per β):
  - `L=16`: `[0.998, 0.968, 0.968, 0.936, 0.906, 0.907, 0.890, 0.972]`
  - `L=32`: `[0.992, 0.900, 0.868, 0.770, 0.621, 0.605, 0.642, 0.888]`
  - `L=64`: `[0.988, 0.784, 0.712, 0.462, 0.178, 0.242, 0.330, 0.778]`
- `results_e :: Dict{Int,Vector{Float64}}` (⟨E⟩/site):
  - `L=16`: `[-0.651,-0.987,-1.036,-1.134,-1.254,-1.369,-1.461,-1.716]`
  - `L=32`: `[-0.677,-1.047,-1.095,-1.220,-1.370,-1.513,-1.598,-1.813]`
  - `L=64`: `[-0.690,-1.074,-1.126,-1.275,-1.455,-1.604,-1.675,-1.862]`
- **Status: valid.** This is acceptance-rate / energy data, not a ρ/C
  measurement, so the ρ-vs-C issue does **not** taint it. Shows the acceptance
  dip near `β_c` worsening with volume — the empirical "fast in practice" puzzle.

### `py/imgs/`
- `critical_temp_acc_rate.png` — acceptance rate vs β near `β_c` (the §6 cell-5/6
  plot, or its Python twin from `critial_point_sim.py`).
- `MHTN1.png`, `MHTN2.png` — algorithm schematics (used as notebook attachments).

**No saved files exist yet for the ρ-tracker / volume-scaling experiments**
(cells 9–22 were re-run live; only the `.png` outputs are embedded in the
notebook). The ρ/"C" numbers therefore have to be regenerated — and **relabeled
as `ρ(Ly,D,β)`** — when revisited (§11).

---

## 8. Conventions & glossary

**Conventions**
- `Lx` = number of rows = **strip length** (sweep direction). `Ly` = number of
  columns = **strip width** (boundary-MPS length). The notebook fixes `Ly` small
  (8–32) so the *exact* environment fits in bond dim `2^{Ly}`.
- `D` = PEPS bond dimension = **2** for Ising (fixed). `D_bound` (a.k.a. `D'`, `χ`)
  = truncation bond dimension of the **boundary MPS** proposal; the knob that
  controls proposal quality. Exact environment uses `maxdim = 2^{Ly}`.
- `β_c = ln(1+√2)/2 ≈ 0.440687`; `T_c = 2/ln(1+√2) ≈ 2.2692` (`J=1`, `k_B=1`).
  Cells 20–21 use the rounded `T_crit=2.269`.
- `J` convention: `H = −J Σ_⟨ij⟩ s_i s_j`, `J=1` ferromagnet. Spins stored `0/1`,
  physically `0→+1`, `1→−1`. Energies in units of `J`; `β` in units of `1/J`.
- Index names: Julia ITensors uses tagged indices `s` (Site/physical), `h`
  (horizontal Link), `v` (vertical Link), `b` (dummy boundary Site for MPS/MPO
  plumbing). Python uses positional axes `(physical, left, up, right, down)` for
  PEPS and `(d_phys, L, R)` / `(d_out, d_in, L, R)` for MPS/MPO.
- Numerical hygiene: truncation `cutoff=1e-10` (proposal) / `1e-15..1e-16`
  (exact); environments renormalized by their max-abs or norm each step (scale
  cancels in conditionals / in the final TV-relevant ratios); log-Z accumulators
  keep absolute probabilities comparable across exact vs truncated branches.

**Glossary** (new-student level)
- **IMH (Independence Metropolis–Hastings):** MH whose proposal `q(x')` ignores
  the current state. Mixing ⇔ a single scalar `C` (§2).
- **Boundary MPS:** the 1D matrix-product state that approximates the
  contraction of all rows on one side of a cut; advanced row-by-row by applying a
  row as an MPO and re-compressing.
- **Bond dimension:** the rank of the virtual index; caps representable
  entanglement (`S ≤ log D`). `D` (PEPS) vs `D_bound` (boundary MPS).
- **SVD / Schmidt truncation:** keep the largest singular values across a cut to
  compress an MPS; discarded weight = approximation error.
- **Transfer matrix:** operator advancing a boundary state by one row/column;
  its dominant eigenvector is the "fixed-point" environment.
- **Fixed-point / plateau:** once the boundary environment converges, per-row
  quantities (like `ρ_x`) stop changing — they *plateau*. (This plateau is `ρ`,
  not `C`.)
- **CFTP (Coupling From The Past):** exact, burn-in-free sampling; for IMH all
  chains coalesce in one step with prob `≥ 1/C`.
- **χ²-divergence:** `χ²(π‖q) = Σ_x q(x)(π/q − 1)² = E_q[w²] − 1`. An
  *average-case* mismatch measure (vs the worst-case `C = max w`). Governs typical
  mixing / variance of importance weights — the proposed pivot in §11.
- **Relaxation time / spectral gap:** `t_rel = 1/γ = C`; `γ = 1 − λ₂ = 1/C` (Liu).
- **Mengersen–Tweedie constant `C`:** `max_x π(x)/q(x)` (§2). The single number
  that controls IMH mixing.
- **ρ (per-layer rate):** the per-row fixed-point version the code actually
  measures; `C ~ ρ^{Lx}` (§2a).

---

## 9. Known issues / open questions

1. **ρ-vs-C conflation (central).** `run_error_tracker.worst_case_ratios`,
   `compute_asymptotic_C` (→ `C_plateaus_*`), and the markdown cells 8/18/22 all
   call the measured plateau "`C`" / "Mengersen–Tweedie constant" /
   "`τ_corr ≤ C`". It is **`ρ(Ly,D,β)`**, a per-layer marginal ratio. A finite
   plateau does **not** prove uniform ergodicity of the full chain. Relabel and
   re-interpret (the volume-scaling plots are `ρ` vs `Ly`, exponential in width).
2. **Marginal vs conditional `[1,1]` cap.** In `run_error_tracker` the measurement
   row's upper vertical bond is tied off with `[1.0,1.0]` (`main.jl`-style), which
   marginalizes everything above the row. The quantity that enters `w(x)=π/q` is
   the **conditional** chain-rule probability of a *full* configuration, not a
   per-row marginal. Decide whether the cap is intended (it gives `ρ`, which is
   the right object for the *per-layer* story but the wrong object for `C`).
   `compute_asymptotic_C` measures at the genuine top row (no cap there) but is
   still a single-row marginal.
3. **No true-`C` measurement in-repo yet.** The claimed brute-force confirmations
   (`log C` linear in `Lx`; `λ₂=1−1/C` by exact diagonalization) are **not present
   as code/artifacts** here — they need to be (re)implemented and saved (§11).
4. **`PEPS_simulation.py` is broken** (`Lx/Ly` undefined; only `L=11`). `NameError`
   on run.
5. **`Z_simulation.py` line 72** — `beta = 0.3` appended to a comment (dead/typo).
6. **`TNMH.py` D_bound mismatch** — init uses `10`, loop uses `8`.
7. **Windows venv committed** (`py/icfo_env/`, `D:\…` paths) — non-portable, won't
   run on this Linux box; should be `.gitignore`d and recreated.
8. **No saved ρ/volume-scaling data** — only embedded plots; not reproducible
   without re-running the (long) sweeps.
9. **Filename typo** `critial_point_sim.py` (missing `c`).
10. **`compute_norm` (quantum PEPS, `D²` layer) is unused** by the classical Ising
    pipeline — dead-ish code kept from the generic PEPS class.

---

## 10. How to run things

### Julia (active codebase)
```bash
cd jl/icfo_intern
julia --project=.            # then: using Pkg; Pkg.instantiate()   (first time)
```
- Versions are pinned in `Manifest.toml` (ITensors 0.9.25, ITensorMPS 0.3.45,
  JLD2 0.6.4, Plots, ProgressMeter). `julia` is available via `~/.juliaup`.
- Run the notebook `simulation.ipynb` (IJulia), or `include("main.jl")` and call
  `run_mcmc_sweep`, then the cell-9/19 functions for ρ trackers.
- **Runtimes (from saved output):** phase sweep 16×16 ≈ 35 min, 32×32 ≈ 2.5 h,
  64×64 ≈ 11 h (`D_bound=2, N=2000`). ρ trackers at `Lx=40, Ly≤16` are minutes;
  the `2^{Ly}` brute-force inner loop and the exact environment make **large
  `Ly` the wall**: the code requests `maxdim = 2^{Ly}` (the true horizontal-cut
  rank saturates at `2^{Ly/2}`), and the measurement loops over all `2^{Ly}`
  row configs (`Ly=20` ⇒ ~10⁶ configs). Keep `Ly ≲ 20` for the exact branch.

### Python (prototype)
```bash
cd py
python3 -m venv .venv && source .venv/bin/activate    # DO NOT use icfo_env/ (Windows)
pip install numpy scipy matplotlib opt_einsum tqdm    # numba optional
python TNMH_v2.py            # IMH demo (4×4)
python critial_point_sim.py # phase-transition sweep (slow at L=64)
python MPS_simulation.py     # unit tests
```
- `py/icfo_env/` is a **Windows** venv (`D:\programs\miniconda3`, py3.13.9) copied
  into the repo; its `Scripts/*.exe` cannot execute on Linux. Recreate as above.
- Gotchas: exact contraction / brute-force `Z` are exponential — keep `L ≤ ~12`
  (contraction) / `N ≤ ~16` (brute force). Fix `PEPS_simulation.py` before running.

---

## 11. Next steps (revised priorities — concrete coding tasks)

1. **Relabel the per-layer rate.** In `simulation.ipynb`: rename
   `worst_case_ratios`→`rho_x`, `compute_asymptotic_C`→`compute_rho_plateau`,
   `C_plateaus_*`→`rho_plateaus_*`, axis/title "C"→"ρ(Ly,D,β)", and rewrite the
   markdown cells 8/18/22 so the plateau is the **per-layer rate** and the
   uniform-ergodicity claim is stated for the *strip layer*, not the full chain.
   Make the `C ~ ρ^{Lx}` relation explicit. Persist outputs to a `.jld2`.
2. **Measure the true full-lattice `C` on small lattices (brute force).** Add a
   routine that enumerates all `2^{Lx·Ly}` configs and computes
   `w(x)=π(x)/q(x)`: `π` from `measure_energy` + exact `Z` (Julia exact env at
   `2^{Ly}`, or Python `compute_Z_brute_force`/`contract_2d_exact`); `q(x)` from
   the **deterministic** chain-rule proposal probability (run the
   `sample_config_opt` conditionals with the spins *pinned* to `x` instead of
   sampled, summing the per-site `log p_chosen`). Then:
   - confirm `log C` grows **linearly in `Lx`** (slope `= log ρ`), and that
     intercept/slope match `ρ` from task 1;
   - build the explicit transition kernel on the `2^{Lx·Ly}` states and check
     `λ₂ = 1 − 1/C` by exact diagonalization (the claimed 5-decimal match).
   Save the brute-force `C`, `ρ`, and `λ₂` arrays.
3. **Pivot to average-case / typical mixing.** The empirical acceptance rates are
   high even though worst-case `C` is exponential in volume at `T_c`. Implement
   average-case diagnostics that explain this:
   - `χ²(π‖q) = E_q[w²] − 1` and the importance-weight distribution (mean/var of
     `w`) — estimable by sampling from `q` (reuse `sample_config_opt`'s `log q`
     + `measure_energy`), no enumeration needed;
   - warm-start / restricted-conductance (typical-set) mixing bounds, i.e.
     restrict the `max` in `C` to a high-probability set and show the *restricted*
     constant is small;
   - relate these to the observed acceptance-rate-vs-β-and-`L` curves already in
     `ising_mcmc_results_D2.jld2`.

---

## 12. Research-plan notebooks (Phases 0–5) — the in-repo answer pipeline

**Status:** primitives **written + unit-tested via CLI** (small lattices); the
notebooks are **not yet run end-to-end** (no heavy `results/*.jld2` yet — those
sweeps are the user's to run). Driven by [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

### `jl/icfo_intern/tools/tnmh_tools.jl` (NEW — the missing primitives)
Assumes `include("main.jl")` first. All validated (see "Validation" below).
- `proposal_logprob(config,Lx,Ly,β,D)` + `proposal_logprob!(…,A,s,v,bottom_envs,…)`
  — **deterministic `log q(config)` for an arbitrary pinned config** (the key
  primitive the repo lacked). Pinned twin of `sample_config_opt`.
- `sample_classical_1d_pinned` — pinned twin of `sample_classical_1d`.
- `enumerate_weights(Lx,Ly,β,D)` — all `2^{N}` configs → `logq, logπ, logw, E,
  w, C=max w, sum_q (Σq check), logZ, argmax`. **First true-`C` path in repo.**
- `imh_kernel(logq,logw)` / `lambda2(P)` / `tv_decay` — IMH kernel + spectrum (Liu).
- `compute_rho_plateau` / `rho_tracker` — clean, correctly-named **ρ** (the §11.1
  relabel, done non-destructively; `simulation.ipynb` left as-is). Marginal `[1,1]`
  cap retained; the conditional ρ is obtained as the `log C`-vs-`Lx` slope instead.
- `truncation_error_fidelity`, `saturated_boundary_mps` — discarded-weight `ε_D`.
- `weight_samples`, `chi2_exact`, `chi2_sampled`, `restricted_C`,
  `integrated_autocorr` — typical-case diagnostics.
- `cftp_coalescence`, `glauber_tau_int` — independent validations.
- helpers: `logsumexp`, `config_from_int` (column-major bit order), `set_seed`.

### Notebooks (Julia unless noted; each saves to `results/` + appends `results/FINDINGS.md`)
- `phase0_infrastructure.ipynb` — primitives demo + controls (**Σq=1**; **C=1 at
  exact D**) + Julia↔Python cross-check (reads `results/crosscheck_py.csv`).
- `py/phase0b_python_crosscheck.ipynb` (Python) — independent `log q` via the new
  `PEPS.proposal_logprob`; writes `results/crosscheck_py.csv`. Needs a real venv
  (the committed `icfo_env/` is Windows; recreate per §10).
- `phase1_spectrum.ipynb` — `λ₂ = 1−1/C` table + TV-decay tightness.
- `phase2_scaling.ipynb` — `ρ(Ly,D,β)` surfaces, `log C` vs `Lx` slope (`ρ_eff`),
  `ρ_eff` vs `ρ_marginal` (resolves §9.2), `ρ` vs `ε_D`, `D*(Ly)`.
- `phase3_typical.ipynb` — weight histogram, `χ²` vs `C` vs volume, restricted
  `C_S`, `τ_int` (ties to `ising_mcmc_results_D2.jld2`).
- `phase4_validation.ipynb` — CFTP coalescence, single-spin-flip Glauber speedup.
- `phase5_synthesis.ipynb` — master figures F1–F5 + the two-part answer.

**Run order:** phase0 (+ phase0b) → phase1 (**gate:** confirm `λ₂=1−1/C`) → 2 → 3 → 4 → 5.

### Validation already passed (CLI only, tiny lattices — NOT from a notebook run)
These were checked by short `julia --project -e` / Python scripts on 2–4-wide
strips; **no notebook has been executed and no result file is saved.** What passed:
- `Σ_x q(x) = 1` to 1e-12; pinned `log q` **==** sampler `log q` (Δ = 0).
- `C = 1` exactly at exact `D` (e.g. 3×4, D=4); `C > 1` once truncation bites (D=2).
- **`λ₂ = 1 − 1/C` to ~6e-15** — Liu's identity confirmed in-repo (the headline
  §9.3 claim, previously only asserted).
- Julia ↔ Python `log q` agree to ~1e-12 on canonical configs.
- All notebook *glue* (Plots/`jldsave`/fits/CSV-IO/reading the existing `.jld2`)
  ran at tiny sizes → no errors. One **benign** `warn_once` fires from ITensorMPS
  on `inner(e,eD)` in `truncation_error_fidelity` (two MPS sharing site indices —
  the overlap is correct; verified `ε_D=0` at exact `D`). Cosmetic, fires once.

### What is NOT done yet (the user's next actions)
- Run the notebooks in order to **produce `results/*.jld2` + `.png` + populate
  `FINDINGS.md`** (currently only the scaffold exists).
- Nothing is git-committed; the throwaway notebook generator and test artifacts
  were deleted (only `tools/tnmh_tools.jl` + the `phase*` notebooks remain).

### Effect on §9 / §11
- §9.3 ("no true-`C` measurement in-repo") and §11.2 — **infrastructure done**
  (`enumerate_weights` + `imh_kernel`); awaiting the user running the sweeps to
  persist data.
- §11.1 (relabel ρ) — realized in `tools/` (clean names) without editing
  `simulation.ipynb`. §11.3 (average-case) — realized in `phase3`.
- The `peps.py` change is **additive** (`proposal_logprob`, `sample_1d_mps_pinned`);
  no behaviour change to existing code.

---

### Quick "what is this number?" lookup

| You see in code/plot | What it really is | Where |
|---|---|---|
| `log_prob` / `log_prob_tot` / `log q` | `log q(config)` (proposal log-prob) | `sample_config_opt` (Julia & Py) |
| `worst_case_ratios`, `C_x` | **ρ_x** per-row marginal ratio | `run_error_tracker`, cell 8/9 |
| `max_ratio`, `C_plateaus_*`, "Plateau Constant C" | **ρ(Ly,D,β)** per-layer rate | `compute_asymptotic_C`, cells 19–21 |
| `fidelity` | `|⟨E_exact|E_trunc⟩|` additive error | `run_error_tracker` |
| `acc_rate`, `results_acc` | empirical MH acceptance | `run_mcmc_sweep`, the `.jld2` |
| true `C` (Mengersen–Tweedie) | `max_x π/q` ≈ `ρ^{Lx}` | **not yet computed** — task §11.2 |
