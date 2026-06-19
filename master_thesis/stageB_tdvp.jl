# Stage B — TDVP Loschmidt amplitude arbiter (run outside the notebook for background execution).
# Corresponds to 13_tdvp_crosscheck.ipynb.
# Run:  julia --project=. stageB_tdvp.jl
using ITensors, ITensorMPS, ITransverse
using Plots, ProgressMeter, LinearAlgebra, JLD2, Printf

include("main.jl")
include("dqpt_diagnostics.jl")

const p      = 0.1
const lambda = 1.0
const dt     = 0.05
const cutoff = 1e-12
const maxdim = 256
const Ns     = [40, 80]  # optimized: drop N=160 for speed
const Ts     = vcat(collect(0.5:0.5:4.0), collect(4.5:0.5:7.0))  # coarser near T=6

@info "Stage B: TDVP Loschmidt amplitude, p=$p, λ=$lambda, dt=$dt, Ns=$Ns, T∈[$(Ts[1]),$(Ts[end])]"

res = Dict{Int,Dict}()
for N in Ns
    @info "===== N=$N ====="
    res[N] = tdvp_loschmidt_amplitude(N, Ts;
        p=p, lambda=lambda, dt=dt, cutoff=cutoff, maxdim=maxdim)
    GC.gc()
end

# ── per-T table (T ∈ [4,7]) ─────────────────────────────────────────────────
println("\n", "="^(6 + 28*length(Ns)))
println("STAGE B — TDVP Loschmidt table (T ∈ [4,7])")
println("="^(6 + 28*length(Ns)))
@printf("%-6s", "T")
for N in Ns; @printf("|   N=%-3d |G|   rate  χ  ", N); end; println()
println("-"^(6 + 28*length(Ns)))
for T in sort(Ts)
    T < 4.0 && continue
    @printf("%-6.2f", T)
    for N in Ns
        r = get(res[N], T, nothing)
        isnothing(r) && (@printf("|       ---            "); continue)
        haskey(r, :error) && (@printf("|  ERROR              "); continue)
        @printf("| %7.5f %7.4f %3d  ", r.absG, r.rate, r.maxchi)
    end
    println()
end

# ── decision ────────────────────────────────────────────────────────────────
println("\n", "="^65)
println("STAGE B DECISION")
println("="^65)
T_mins = Dict{Int,Tuple{Float64,Float64}}()
for N in Ns
    haskey(res, N) || continue
    Tv = sort(collect(keys(res[N])))
    Gv = [res[N][T].absG for T in Tv]
    idx = argmin(Gv)
    T_mins[N] = (Tv[idx], Gv[idx])
    @printf("  N=%3d: |L| minimum at T=%.2f  (|L|=%.5f)\n", N, Tv[idx], Gv[idx])
end

Ns_s     = sort(collect(keys(T_mins)))
Ts_star  = [T_mins[N][1] for N in Ns_s]
Gs_star  = [T_mins[N][2] for N in Ns_s]
T_spread = maximum(Ts_star) - minimum(Ts_star)
deepening  = Gs_star[end] < Gs_star[1]
consistent = T_spread < 0.6
@printf("\n  T* spread: %.2f  consistent: %s\n", T_spread, consistent)
@printf("  |L| deepening with N: %s\n\n", deepening)

if consistent && deepening
    T_star = round(sum(Ts_star)/length(Ts_star), digits=2)
    println("VERDICT: GENUINE DQPT near T* ≈ $T_star. Proceed to Stage C.")
elseif !consistent
    println("VERDICT: INCONCLUSIVE — T* drifts with N.")
else
    println("VERDICT: LIKELY ARTIFACT — T* consistent but |L| not deepening.")
end
println("="^65)

# ── figure ───────────────────────────────────────────────────────────────────
mkpath("imgs")
cols = Dict(40=>:seagreen, 80=>:royalblue)
ls_  = Dict(40=>:solid,    80=>:dash)

p1 = plot(title="|L(T)| — Alcaraz p=$p", xlabel="T", ylabel="|L(T)|",
          grid=true, framestyle=:box, legend=:topright)
p2 = plot(title="Rate l(T)=-log|L|/N",  xlabel="T", ylabel="l(T)",
          grid=true, framestyle=:box, legend=:topleft)
p3 = plot(title="arg L(T)",              xlabel="T", ylabel="arg L (rad)",
          grid=true, framestyle=:box, legend=:topright)

for N in Ns
    haskey(res, N) || continue
    Tv = sort(collect(keys(res[N])))
    plot!(p1, Tv, [res[N][T].absG   for T in Tv], lw=2, color=cols[N], ls=ls_[N], ms=3, marker=:circle, label="N=$N")
    plot!(p2, Tv, [res[N][T].rate   for T in Tv], lw=2, color=cols[N], ls=ls_[N], ms=3, marker=:circle, label="N=$N")
    plot!(p3, Tv, [angle(res[N][T].G) for T in Tv], lw=2, color=cols[N], ls=ls_[N], ms=3, marker=:circle, label="N=$N")
end
for pp in [p1,p2,p3]; vline!(pp, [6.0], ls=:dash, color=:gray, alpha=0.5, label="T=6"); end
hline!(p3, [π,-π], ls=:dot, color=:black, alpha=0.4, label="±π")

fig = plot(p1, p2, p3; layout=(1,3), size=(2000,480),
           plot_title="Stage B TDVP — Alcaraz p=$p, dt=$dt", margin=5Plots.mm)
savefig(fig, "imgs/stageB_tdvp_arbiter_p$(p)_dt$(dt).png")
println("Figure saved: imgs/stageB_tdvp_arbiter_p$(p)_dt$(dt).png")
