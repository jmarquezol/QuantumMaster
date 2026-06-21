using ITensors, ITensorMPS, LinearAlgebra

function finitestatemachine(
  ValType::Type{<:Number}, os::OpSum{C}, sites; mindim, maxdim, cutoff
) where {C}
  # This function takes an OpSum and the sites and returns a representation of the MPO as a finite state machine.
  N = length(sites)

  # Specifying the element type with `Matrix{ValType}[...]` improves type inference and therefore efficiency.
  # See https://github.com/ITensor/ITensorMPS.jl/pull/1183.
  Vs = Matrix{ValType}[Matrix{ValType}(undef, 1, 1) for n in 1:N]
  tempMPO = [ITensorMPS.MatElem{Scaled{C,Prod{Op}}}[] for n in 1:N]

  # helper function to check if a term in the OpSum crosses the bond between sites n and n+1
  function crosses_bond(t::Scaled{C,Prod{Op}}, n::Int) where {C}
    return (only(ITensors.site(t[1])) <= n <= only(ITensors.site(t[end])))
  end

  rightmaps = [Dict{Vector{Op},Int}() for _ in 1:N]

  # this function computes a representation of the MPO as a finite state machine.
  for n in 1:N
    leftbond_coefs = ITensorMPS.MatElem{ValType}[]

    leftmap = Dict{Vector{Op},Int}()
    for term in os
      crosses_bond(term, n) || continue # if the term does cross the bond, then it splits into three parts:

      left = filter(t -> (only(ITensors.site(t)) < n), ITensors.terms(term))
      onsite = filter(t -> (only(ITensors.site(t)) == n), ITensors.terms(term))
      right = filter(t -> (only(ITensors.site(t)) > n), ITensors.terms(term))

      bond_row = -1
      bond_col = -1
      if !isempty(left)
        bond_row = ITensorMPS.posInLink!(leftmap, left)
        bond_col = ITensorMPS.posInLink!(rightmaps[n - 1], vcat(onsite, right))
        bond_coef = convert(ValType, ITensorMPS.coefficient(term))
        push!(leftbond_coefs, ITensorMPS.MatElem(bond_row, bond_col, bond_coef))
      end

      A_row = bond_col
      A_col = ITensorMPS.posInLink!(rightmaps[n], right)
      site_coef = one(C)
      if A_row == -1
        site_coef = ITensorMPS.coefficient(term)
      end
      if isempty(onsite)
        if !ITensorMPS.using_auto_fermion() && ITensorMPS.isfermionic(right, sites)
          push!(onsite, Op("F", n))
        else
          push!(onsite, Op("Id", n))
        end
      end
      el = ITensorMPS.MatElem(A_row, A_col, site_coef * Prod(onsite))
      push!(tempMPO[n], el)
    end
    ITensorMPS.remove_dups!(tempMPO[n])
    if n > 1 && !isempty(leftbond_coefs)
      # compression step to find minimum number of memory channels needed to carry the interactions
      M = ITensorMPS.toMatrix(leftbond_coefs)
      U, S, V = svd(M)
      P = S .^ 2
      truncate!(P; maxdim=maxdim, cutoff=cutoff, mindim=mindim)
      tdim = length(P)
      nc = size(M, 2)
      Vs[n - 1] = Matrix{ValType}(V[1:nc, 1:tdim])
    end
  end

  return tempMPO, Vs, rightmaps
end

function eulermpo(
  dt, os::OpSum{X}, sites, alg::Algorithm; mindim=1, maxdim=typemax(Int), cutoff=1e-15
)::MPO where {X}
  ElT = promote_type(typeof(dt), X)

  tempMPO, Vs, rightmaps = finitestatemachine(ElT, os, sites; mindim, maxdim, cutoff)

  N = length(sites)
  llinks = Vector{Index{Int}}(undef, N + 1)
  llinks[1] = Index(1, "Link,l=0")

  U = MPO(sites)

  for n in 1:N
    VL = n > 1 ? Vs[n - 1] : Matrix{ElT}(undef, 0, 0)
    VR = Vs[n]
    tdim = isempty(rightmaps[n]) ? 0 : size(VR, 2)

    # modification:
    if alg isa Algorithm"VD2"
        llinks[n + 1] = Index(1 + tdim + tdim^2, "Link,l=$n")
    else
        llinks[n + 1] = Index(1 + tdim, "Link,l=$n")
    end

    ll = llinks[n]
    rl = llinks[n + 1]

    ri = Index(dim(ll) - 1, "n,at=$n")
    ci = Index(dim(rl) - 1, "p,at=$n")
    s = sites[n]

    A = ITensor(ElT, ri, ci, s, s')
    B = ITensor(ElT, ri, s, s')
    C = ITensor(ElT, ci, s, s')
    D = ITensor(ElT, s, s')

    A .= zero(ElT)
    B .= zero(ElT)
    C .= zero(ElT)
    D .= zero(ElT)

    U[n] = ITensor()

    for el in tempMPO[n]
      A_row = el.row
      A_col = el.col
      t = el.val
      (abs(coefficient(t)) > eps()) || continue

      @assert length(ITensorMPS.argument(t)) == 1

      mat = op(s, ITensorMPS.which_op(ITensorMPS.argument(t)[1]))

      ct = convert(ElT, coefficient(t))
      if A_row == -1 && A_col == -1
        # onsite term => D
        D += ct * mat
      elseif A_row == -1
        # term starting on site n => C
        for c in axes(VR, 2)
          z = ct * VR[A_col, c]
          C += z * onehot(ci => c) * mat
        end
      elseif A_col == -1
        # term ending on site n => B
        for r in axes(VL, 2)
          z = ct * conj(VL[A_row, r])
          B += z * onehot(ri => r) * mat
        end
      else
        # belongs to A
        for r in axes(VL, 2), c in axes(VR, 2)
          z = ct * conj(VL[A_row, r]) * VR[A_col, c]
          A += z * onehot(ci => c, ri => r) * mat
        end
      end

      # this function takes the coefficients M (ll, rl) and attaches at each entry the respective matrix argument(t)
    end

    W = makeW(alg, ElT, dt, A, B, C, D)

    U[n] = ITensorMPS.itensor(W, ll, rl, sites[n], sites[n]')
  end

  L = ITensor(llinks[1])
  L[end] = 1.0

  R = ITensor(llinks[N + 1])
  R[1] = 1.0

  U[1] *= L
  U[N] *= R

  return U
end