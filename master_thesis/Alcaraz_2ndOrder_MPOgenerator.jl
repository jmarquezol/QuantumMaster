using ITensors, ITensorMPS
using Combinatorics, LinearAlgebra
using ITransverse: ModelParams, modelparams

""" Builds the optimal 2nd-order MPO Generator for the Alcaraz model (WII construction) """

struct AlcarazModel <: ModelParams
    lambda::Float64
    p::Float64
    phys_site::Index
end

ITransverse.modelparams(mp::AlcarazModel) = (mp.lambda, mp.p)

# HELPER FUNCTIONS

# Computes the Direct Product on virtual legs and Matrix Product on physical legs
function otimes(X::Matrix{Matrix{ComplexF64}}, Y::Matrix{Matrix{ComplexF64}})
    nx, mx = size(X)
    ny, my = size(Y)
    
    Z = Matrix{Matrix{ComplexF64}}(undef, nx * ny, mx * my)
    
    for ix in 1:nx, jx in 1:mx
        for iy in 1:ny, jy in 1:my
            row = (ix - 1) * ny + iy
            col = (jx - 1) * my + jy
            Z[row, col] = X[ix, jx] * Y[iy, jy] # Physical matrix multiplication
        end
    end
    return Z
end

# Computes the symmetric sum over all unique permutations of a set of operators
function sym_sum(ops::Vector{Matrix{Matrix{ComplexF64}}})
    perms = unique(permutations(ops))
    
    res = nothing
    for p in perms
        term = p[1]
        for k in 2:length(p)
            term = otimes(term, p[k])
        end
        
        if res === nothing
            res = copy(term)
        else
            res .+= term
        end
    end
    return res
end


# OPTMIAL 2ND ORDER MPO GENERATOR

function expH_alcaraz_optimal_2nd(sites::Vector{<:Index}, lambda::Number, p::Number; dt::Number)
    N = length(sites)
    
    I2 = ComplexF64[1 0; 0 1]
    Z  = ComplexF64[1 0; 0 -1]
    X  = ComplexF64[0 1; 1 0]
    O2 = ComplexF64[0 0; 0 0]
    
    tau = -im * dt 

    # BASE HAMILTONIAN BLOCKS:

    D = Matrix{Matrix{ComplexF64}}(undef, 1, 1)
    D[1, 1] = -lambda .* X
    
    C = Matrix{Matrix{ComplexF64}}(undef, 1, 3)
    C[1, 1] = Z; C[1, 2] = O2; C[1, 3] = X
    
    B = Matrix{Matrix{ComplexF64}}(undef, 3, 1)
    B[1, 1] = -Z; B[2, 1] = -p .* Z; B[3, 1] = -p * lambda .* X
    
    A = Matrix{Matrix{ComplexF64}}(undef, 3, 3)
    for i in 1:3, j in 1:3
        A[i, j] = O2
    end
    A[1, 2] = I2
    
    I_op = Matrix{Matrix{ComplexF64}}(undef, 1, 1)
    I_op[1, 1] = I2

    # Using formulasfrom Appendix A of Van Damme et al. 2023:
    
    # Macro-Row 1
    W11 = I_op .+ tau .* D .+ (tau^2 / 2) .* otimes(D, D) .+ (tau^3 / 6) .* otimes(D, otimes(D, D))
    W12 = C .+ (tau / 2) .* sym_sum([C, D]) .+ (tau^2 / 6) .* sym_sum([C, D, D])
    W13 = otimes(C, C) .+ (tau / 3) .* sym_sum([C, C, D])

    # Macro-Row 2
    W21 = tau .* B .+ (tau^2 / 2) .* sym_sum([B, D]) .+ (tau^3 / 6) .* sym_sum([B, D, D])
    W22 = A .+ (tau / 2) .* (sym_sum([B, C]) .+ sym_sum([A, D])) .+ (tau^2 / 6) .* (sym_sum([C, B, D]) .+ sym_sum([A, D, D]))
    W23 = sym_sum([A, C]) .+ (tau / 3) .* (sym_sum([A, C, D]) .+ sym_sum([C, C, B]))

    # Macro-Row 3
    W31 = (tau^2 / 2) .* otimes(B, B) .+ (tau^3 / 6) .* sym_sum([B, B, D])
    W32 = (tau / 2) .* sym_sum([A, B]) .+ (tau^2 / 6) .* (sym_sum([A, B, D]) .+ sym_sum([B, B, C]))
    W33 = otimes(A, A) .+ (tau / 3) .* (sym_sum([A, B, C]) .+ sym_sum([A, A, D]))

    # Assemble full 13x13 block matrix
    W_opt = hvcat((3, 3, 3), W11, W12, W13, W21, W22, W23, W31, W32, W33)
    
    # Build the ITensor MPO (new bond dimension is 1 (W11) + 3 (W22) + 9 (W33) = 13)
    dim_opt = 13
    links = [Index(dim_opt, "Link,l=$i") for i in 1:N-1]
    U_dt = MPO(sites)
    
    for i in 1:N
        s = sites[i]
        
        if i == 1
            W = ITensor(ComplexF64, links[1], s', s)
            for col in 1:dim_opt, s1 in 1:2, s2 in 1:2
                W[links[1]=>col, s'=>s1, s=>s2] = W_opt[1, col][s1, s2]
            end
            U_dt[i] = W
            
        elseif i == N
            W = ITensor(ComplexF64, links[i-1], s', s)
            for row in 1:dim_opt, s1 in 1:2, s2 in 1:2
                W[links[i-1]=>row, s'=>s1, s=>s2] = W_opt[row, 1][s1, s2]
            end
            U_dt[i] = W
            
        else
            W = ITensor(ComplexF64, links[i-1], links[i], s', s)
            for row in 1:dim_opt, col in 1:dim_opt, s1 in 1:2, s2 in 1:2
                W[links[i-1]=>row, links[i]=>col, s'=>s1, s=>s2] = W_opt[row, col][s1, s2]
            end
            U_dt[i] = W
        end
    end
    
    return U_dt
end

