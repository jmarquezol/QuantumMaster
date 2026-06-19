# Stage A — convergence triage for the Alcaraz transverse power method.
# Question: at T≳5, is powermethod_lr stuck on TOLERANCE (ΔS still decaying, would
# converge with more iters) or on a GENUINE GAP CLOSING (ΔS plateaus above eps)?
#
# F2 (see plan): the PM never reaches itermax; it stops on stuck_after=200. So we
# DISABLE the early stop (stuck_after = itermax) and watch the full ΔS trajectory.
# Run:  julia --project=. stageA_triage.jl
using ITensors, ITensorMPS, ITransverse
using Plots, ProgressMeter, LinearAlgebra, JLD2, Printf

include("main.jl")
include("dqpt_diagnostics.jl")

const p       = 0.1
const lambda  = 1.0
const dt      = 0.1
const maxdim  = 256
const ITERMAX = 1000          # hard runtime bound
const Ts      = [5.0, 6.0, 7.0]
const cache   = "stageA_triage_p$(p).jld2"

done = isfile(cache) ? load(cache, "done") : Dict{Float64,Any}()

for T in Ts
    haskey(done, T) && (println("T=$T cached, skip"); continue)
    @info "Stage A triage: T=$T (stuck_after disabled, itermax=$ITERMAX)"
    r = run_pm_diagnosed(T; p=p, lambda=lambda, dt=dt, maxdim=maxdim,
                         eps_converged=1e-6, nbeta=0, MPO_alg="VD2",
                         itermax=ITERMAX, stuck_after=ITERMAX)
    done[T] = (niters=r.niters, reason=r.reason, final_ds=r.final_ds,
               eps_converged=r.eps_converged, itermax=r.itermax,
               lambda0=r.lambda0, lr_cos=r.lr_cos,
               ds_hist=r.ds_hist, chi_hist=r.chi_hist)
    jldsave(cache; done)
    GC.gc()
end

# ── table + trajectory diagnosis ────────────────────────────────────────────
println("\n", "="^78)
println("STAGE A — convergence triage table (p=$p, dt=$dt, maxdim=$maxdim)")
println("="^78)
@printf("%-5s %7s %-10s %11s %11s  %-12s\n",
        "T","niters","reason","final_ds","eps_conv","ds trend")
for T in sort(collect(keys(done)))
    r = done[T]
    ds = r.ds_hist
    # trend: compare median of last 20% of iters to median of the 20% before stuck cutoff
    n = length(ds)
    trend = if n < 50
        "too short"
    else
        tail   = ds[max(1, n - n÷5 + 1):n]
        midwin = ds[max(1, n - 2*(n÷5) + 1):(n - n÷5)]
        mt, mm = sum(tail)/length(tail), sum(midwin)/length(midwin)
        mt < 0.5*mm ? "DECAYING" : "PLATEAU"
    end
    @printf("%-5.1f %7d %-10s %11.3e %11.1e  %-12s\n",
            T, r.niters, r.reason, r.final_ds, r.eps_converged, trend)
end
println("="^78)
println("DECISION: 'DECAYING' toward eps ⇒ tolerance-limited (fixable). ")
println("          'PLATEAU' above eps ⇒ genuine gap closing ⇒ Stage C mandatory.")
