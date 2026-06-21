# CONTINUATION — session handoff (read this first)

> Handoff for a fresh Claude session. Read this, then `CLAUDE.md` for the full technical context.
> Written after the session that remade NB7, added NB3.5, and added a dt-appendix to NB3.

---

## 1. Status snapshot

Thesis repo is the clean **1–7 (+3.5)** notebook series with a consolidated `src/` library. The live
research thread: **does the temporal central charge stay at `c = 1/2` for the non-integrable Alcaraz
model (`p = 0.1`)?** The framework is validated end-to-end on the integrable Ising chain (NB6); the
`p = 0.1` extraction is blocked mainly by the **cost and convergence of the block power method**,
which is the #1 thing to fix next.

---

## 2. What this session changed

- **Library `src/transverse_tools.jl`** — `block_transfer_eigs` gained two **backward-compatible**
  kwargs (defaults reproduce the old behavior exactly):
  - `maxdims` — per-iteration bond-dim **RAMP** (cheap early iters, then hold the cap).
  - `seedL`/`seedR` — **warm-start** seeds (padded with random if fewer than `k`).
  - `compute_entropies` forwards `maxdims`.
- **NB1** — real Trotter-gate TEBD vs cached TDVP `⟨Z⟩` benchmark (the curves *agree well*; the
  problem with the naive route is **structural** — global `apply` breaks translation invariance — not
  accuracy). Fixed duplicate-cell scrambles.
- **NB3** — block-PM convergence narrative + `use_block_pm=true`; **additive dt-appendix**
  (δt=0.05 nbeta=8 vs δt=0.1 nbeta=4, single-vector first then block PM). *Not yet run.*
- **NB4** — restored the DMRG finite-size scaling + chord fits that had been dropped.
- **NB5** — added "the gap closes faster at larger p".
- **NB6** — λ₀ circle for emergent dual unitarity; λ₀ now from the **cheap symmetric Rayleigh
  quotient** (not block PM). Radius ≈ 1.497 ≠ 1 is the non-universal tMPO normalization — the
  *constancy* of the radius across T is the physics, not its value.
- **NB7** — **completely remade**: two routes — (1) Rényi-2 entropy slope `c = 8·slope`; (2) leading
  eigenvalues (Eq. 3 / Eq. 4 + the λ₀ circle) — with `p = 0` as a convention calibration. Split into a
  **lite** sweep (eigenvalues only, cheap) and a **heavy** sweep (entropy profiles).

---

## 3. Key results this session (the actual numbers)

### NB7 Route 2 — lite sweep (k=2, itermax=400, nbeta=4)
- **λ₀ circle:** `|Λ₀| ≈ 1.49` (p=0), `≈ 1.55` (p=0.1), **constant radius at both** ⇒ emergent dual
  unitarity **survives the NNN frustration**. ✅
- **Eq. (4) boundary exponent:** **p=0 textbook** — `x₁ = 0.498`, `a₁ = 0.782 ≈ π/4` ⇒ free-BC Ising
  confirmed, the eigenvalue route is validated. **p=0.1 broken** — `x₁ = 0.776`, `a₁` sign-flipped.
- **Eq. (3):** p=0 `a₂ = −0.026`, `c_nominal = 0.99 ≈ 2 × ½` (prefactor convention off by ~2). p=0.1
  `a₂ = −8.5` (blown up) ⇒ `c = 164`. **Broken.**
- **Gap ratio** `|Λ₁|/|Λ₀| → 1`, **faster for p=0.1** (0.999 already by T=4, vs p=0 at T=6). **T=8, 10
  did NOT converge** (`reason = maxiter`) for both p — and those corrupted points enter the Eq.3/Eq.4
  fits, which is *why* p=0.1 looks broken (convergence/noise, not physics).
- **Cross-check:** lite k=2 `|Λ₀|(T=6) = 1.54943` == NB3.5 k=4 reference `1.549433` ⇒ where it
  converges, the eigenvalue is trustworthy.

### NB3.5 — block-PM cost (reference: p=0.1, T=6, nbeta=4 → 64 time sites, VD2 temporal phys-dim 7)
- `k=4, maxdim=256, cutoff=1e-12`: **27 201 s ≈ 7.5 hours**, 177 iters, ~154 s/iter, **converged χ =
  44** ⇒ the cap is **~6× oversized**.
- Route A `maxdim=48`: **4 666 s (~5.8× faster)**, *identical* `|Λ₀| = 1.54943`, same χ = 44.
  (Routes B–F not yet run; the sweep was interrupted at maxdim=64.)

---

## 4. Open problems (prioritized)

1. **Block PM is too expensive** (7.5 h/point) — the bottleneck of the entire p>0 program.
2. **NB7 Route 2 p=0.1 phases are unusable** until the high-T points converge.
3. NB7 Route 1 (entropy profiles) heavy sweep **not yet run**.
4. NB3 dt-appendix **not yet run** — is `T_crit` physical or a δt artifact?
5. Eq. (3) **factor-2 prefactor** (`c_nominal(p=0) = 0.99`, not 0.5) unresolved.

---

## 5. Continuation plan

### PRIORITY 1 — make the block power method much cheaper

Finish NB3.5 and fold the wins into the production sweeps. **Draw directly on ITransverse's own power
method** — read:
`~/.julia/packages/ITransverse/8pmYI/src/power_method/pm.jl` and `.../pm_params.jl`,
`.../truncation_sweeps/trunclr_apply.jl`, `.../ITenUtils/apply_contract.jl`.

- **(a) Lower the maxdim cap.** Confirmed: converged χ = 44, so the cap should be ≈ 64 (not 256).
  Already ~5.8× and it must be applied **everywhere** (lite + heavy sweeps).
- **(b) Per-iteration bond-dim RAMP.** ITransverse's `powermethod_op` does exactly this:
  `maxdim = get(maxdims, jj, maxdims[end])` (pm.jl ~line 22), with the PMParams default
  `maxdims = 2:2:truncp.maxdim` (pm_params.jl line 7). **NOTE:** the LR variant we use,
  `powermethod_lr` (~line 100), does **NOT** ramp — it uses a fixed `truncp`. Our `block_transfer_eigs`
  now has the `maxdims` kwarg (added this session); wire a `2:2:64`-style ramp into the NB7/NB3 sweeps
  and benchmark vs fixed (NB3.5 Route B).
- **(c) Per-iteration CUTOFF schedule** (`cutoffs`, scheduled like `maxdims`) — looser early, tighten
  late (NB3.5 Route C; expect `1e-10` ≈ free).
- **(d) RTM-aware truncation — the deepest structural win.** `powermethod_lr` truncates via
  `tlrapply` (`truncation_sweeps/trunclr_apply.jl`, the `alg="RTM"` route), which truncates on the
  **transition-matrix** structure. Our `block_transfer_eigs` uses the **naive** `applyn`+SVD
  (`apply_contract.jl:19`). Replacing the apply/truncate inside the block method with a tlrapply-style
  RTM truncation should be **both faster** (keeps fewer states) **and more accurate** at the gap
  closing. Investigate this thoroughly — it is the highest-leverage idea.
- **(e) Stuck / itermin / stepper early-stop** — adopt the `pm_itercheck!` / `stuck_after` / `itermin`
  logic so runs stop as soon as the leading θ stops improving, instead of grinding to `itermax`.
- **(f) Warm-start across T.** `seedL`/`seedR` already added (proven same-T in NB3.5 Route E). Build a
  `pad_tmps` helper to move converged T-vectors onto the longer (T+ΔT) scaffold's site indices, then
  seed along the T-ladder — the largest **iteration-count** saving for a full sweep.
- **(g) Finish NB3.5 routes B–F** at the cheap config and write the recommended config into the
  Conclusion cell.

### PRIORITY 2 — fix NB7 Route 2 for p=0.1

- Re-run the lite sweep with the PRIORITY-1 cheap config (`maxdim ≈ 64` + ramp + `cutoff = 1e-10` +
  `k = 2`) and a **much larger `itermax`** so T=8, 10 converge for p=0.1.
- Fit Eq. (3)/Eq. (4) on the **converged-T window only** (drop `maxiter` points); extend the ladder
  (T=12, 14) once they converge.
- Resolve the **factor-2 prefactor**: Eq. (3) p=0 gives `c_nominal = 0.99` (≈ 2×½). Check whether the
  transfer eigenvalue advances **2 time steps per application** (⇒ effective δt → 2δt) or
  `κ = −πcδt/3` in our normalization. The **calibrated ratio** (p=0.1 / p=0) is the safe headline
  regardless of this prefactor.

### PRIORITY 3 — the rest, now affordable

- **NB7 Route 1:** run the heavy sweep with the cheap config; read `c = 8·slope` and the p=0.1/p=0
  ratio.
- **NB3 dt-appendix:** run it — is `T_crit(δt=0.05) == T_crit(δt=0.1)` (physical barrier) or shifted
  (discretization artifact)?

---

## 6. Gotchas / do-not-break (carried from CLAUDE.md)

- Transverse overlaps are **`overlap_noconj`**, never `inner` (the bra is already a bra).
- PM seed must be **random** complex tensors (the structured `fw_tMPS` traps in a subdominant sector).
- `mps ./ scalar` **broadcasts** over all N tensors (scales the norm by `scalarᴺ`) — use scalar mult
  or `normalize`.
- MPS sums: `+(...; alg="directsum")` then `truncate!`; never the density-matrix `+` (NaN on
  degenerate sums). Generalized eig: `pinv(S)*M`, never `eigen(M,S)`.
- `nbeta = 2β₀/δt` (β₀=0.2 ⇒ nbeta=4 at δt=0.1, **8 at δt=0.05**).
- Library edits to `block_transfer_eigs` must stay **backward-compatible** (defaults = old behavior) —
  a long NB7 sweep may be running.

---

## 7. File map

- **Library:** `src/transverse_tools.jl` (`block_transfer_eigs`, `compute_entropies`,
  `run_pm_diagnosed`, `build_alcaraz_tmpo`), `src/models.jl`, entry point `src/thesislib.jl`.
- **Notebooks:** `3_temporal_entropies.ipynb` (dt-appendix), `3.5_block_pm_efficiency.ipynb`
  (routes A–F), `7_loschmidt_alcaraz.ipynb` (lite cell id `9a34810f`, heavy cell id `8fb0be39`,
  Route-2 analysis cells: circle / Eq.4 / Eq.3 / verdict).
- **Caches:** `results/data/nb7_alcaraz_lite.jld2`, `nb7_alcaraz_block.jld2`,
  `nb35_blockpm_bench.jld2`.
- **ITransverse reference:** `~/.julia/packages/ITransverse/8pmYI/src/power_method/{pm,pm_params}.jl`,
  `.../truncation_sweeps/trunclr_apply.jl`, `.../ITenUtils/apply_contract.jl`.
