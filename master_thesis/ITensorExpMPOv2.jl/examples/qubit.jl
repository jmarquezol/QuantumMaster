using ITensors, ITensorMPS, ITensorExpMPO
using ITensors: Algorithm

function main(N=4)
  sites = siteinds("Qubit", N)

  os = OpSum()
  for i in 1:(N - 2)
    os += 1, "S+", i, "S-", i + 1, "Sz", i + 2
  end
  for i in 1:(N - 3)
    os += 1, "Sx", i, "Sz", i + 3
  end
  for i in 1:N
    os += 1, "Sy", i
  end

  ψ = random_mps(sites)

  ψ1 = deepcopy(ψ)
  ψ2 = deepcopy(ψ)
  ψ3 = deepcopy(ψ)

  dt = -0.05im

  WI = expmpo(os, sites, dt; alg=Algorithm("WI"))
  WII = expmpo(os, sites, dt; alg=Algorithm("WII"))
  H = MPO(os, sites)
  maxdim = 100
  cutoff = 1e-8

  tsteps = 100
  for _ in 1:tsteps
    ψ1 = apply(WI, ψ1; maxdim, cutoff)
    ψ2 = apply(WII, ψ2; maxdim, cutoff)
    ψ3 = tdvp(H, dt, ψ3; maxdim, cutoff)

    ψ1 /= norm(ψ1)
    ψ2 /= norm(ψ2)
    ψ3 /= norm(ψ3)

    @show inner(ψ, ψ3), inner(ψ1, ψ3), inner(ψ2, ψ3)
  end
end