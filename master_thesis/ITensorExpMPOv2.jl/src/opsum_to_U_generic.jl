using ITensors: Algorithm, @Algorithm_str, hasqns
using ITensorMPS: OpSum, sorteachterm, sortmergeterms


function expmpo(os::OpSum{X}, sites, dt; alg::Algorithm=Algorithm"WII"(), mindim=1, maxdim=typemax(Int), cutoff=1e-15) where {X}
    os = deepcopy(os)
    os = sorteachterm(os, sites)
    os = sortmergeterms(os)
  
    return if hasqns(first(sites))
        eulermpo_qn(dt, os, sites, alg; mindim, maxdim, cutoff)
    else
        eulermpo(dt, os, sites, alg; mindim, maxdim, cutoff)
    end
end