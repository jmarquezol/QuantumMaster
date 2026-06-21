using ITensors
using ITensors: Algorithm, @Algorithm_str
using LinearAlgebra

function makeW(::Algorithm"WI", ElT, t, A, B, C, D)
    # See: https://github.com/tenpy/tenpy/blob/main/tenpy/networks/mpo.py#L1499
    # W_I \approx I + t W_H
    tC = sqrt(abs(t))
    tB = t / tC
    
    s = commonind(C, D)
    # # We find the virtual memory indices linking the blocks
    ri = noncommonind(A, C) # row index of A
    ci = noncommonind(A, B) # column index of A
    
    d = dim(s)  # physical dimension
    #The virtual size of W is  (1+Nr, 1+Nc)
    Nr = dim(ri)    # number of incoming memory channels (past)
    Nc = dim(ci)    # number of outgoing memory channels (future)
    W = zeros(ElT, 1 + Nr, 1 + Nc, d, d)

    W[1,1,:,:] = I(d) + t * Array(D, s, s') # the '+ 1' makes room for the Identity track (the first row/col of W_H)

    for r in eachval(ri), c in eachval(ci)
        W[1+r, 1+c, :, :] = Array(A * onehot(dag(ci) => c, dag(ri) => r), s, s')
    end
    for c in eachval(ci)
        W[1, 1+c, :, :] = Array(tC * (C * onehot(dag(ci) => c)), s, s')
    end
    for r in eachval(ri)
        W[1+r, 1, :, :] = Array(tB * (B * onehot(dag(ri) => r)), s, s')
    end
    
    W
end

function makeW(::Algorithm"WII", ElT, t, A, B, C, D)
    # See: https://github.com/tenpy/tenpy/blob/main/tenpy/networks/mpo.py#L1499
    # To get $O(t^2)$ precision, we need to square the Hamiltonian. But squaring $W_H$ explicitly creates a mess of cross-terms (like $CD + DC$). 
    # Zaletel solved this elegantly by mapping the blocks to a fictitious quantum system with two "auxiliary bosons" ($a$ and $\bar{a}$).
    tC = sqrt(abs(t))  #spread time step across B, C
    tB = t / tC
    
    s = commonind(C, D)
    ri = noncommonind(A, C)
    ci = noncommonind(A, B)
    
    d = dim(s)
    #The virtual size of W is  (1+Nr, 1+Nc)
    Nr = dim(ri)
    Nc = dim(ci)

    W = zeros(ElT, 1 + Nr, 1 + Nc, d, d)

    # construct indices for the two auxiliary bosons 
    # They have dimension 2 (they can be empty |0> or occupied |1>)
    i1 = Index(2, "first boson") # a
    i2 = Index(2, "second boson") # abar

    # `cd1` is the creation operator a^dag. It maps state 1 (empty) to state 2 (full).
    # `cd2` is \bar{a}^dag.
    cd1 = onehot(ElT, i1 => 1, i1' => 2) # cdag_a
    cd2 = onehot(ElT, i2 => 1, i2' => 2) # cbardag_abar

    # vacuum state |0,0>
    ket00 = onehot(ElT, i1 => 1, i2 => 1)

    # projection bras: <0,0|, <0,1|, <1,0|, <1,1|
    bra00 = onehot(ElT, i1' => 1, i2' => 1)
    bra01 = onehot(ElT, i1' => 1, i2' => 2)
    bra10 = onehot(ElT, i1' => 2, i2' => 1)
    bra11 = onehot(ElT, i1' => 2, i2' => 2)

    # build terms of the effective Hamiltonian: $H_{eff} = D + B a^\dagger + C \bar{a}^\dagger + A a^\dagger \bar{a}^\dagger$
    Id = delta(ElT, i1', i1) * delta(ElT, i2', i2)
    Br = cd1 * delta(ElT, i2', i2)  # Br = a^dag (attached to B)
    Bc = delta(ElT, i1', i1) * cd2  # Bc = \bar{a}^dag (attached to C)
    Brc = cd1 * cd2                 # Brc = a^dag \bar{a}^dag (attached to A)

    for r in eachval(ri)  #double loop over row / column of A
        for c in eachval(ci)
            # 1. Build the effective Hamiltonian for this specific memory channel
            h = Brc * (A * onehot(ElT, dag(ci) => c, dag(ri) => r)) + Br * tB * (B * onehot(ElT, dag(ri) => r)) + Bc * tC * (C * onehot(ElT, dag(ci) => c)) + t * Id * D

            # 2. Exponentiate it and apply it to the vacuum state |0,0>
            w = exp(h) * ket00

            # 3. Project out the resulting blocks using the bras
            # W[1+r, 1+c] corresponds to the state where BOTH bosons were created <1,1|
            W[1+r, 1+c, :, :] = Array(bra11 * w, s, s')
            if c == 1
                W[1+r, 1, :, :] = Array(bra10 * w, s, s')   # W[1+r, 1] corresponds to only the first boson <1,0|
            end
            if r == 1
                W[1, 1+c, :, :] = Array(bra01 * w, s, s')   # W[1, 1+c] corresponds to only the second boson <0,1|
                # W[1, 1] corresponds to no bosons created <0,0|
                if c == 1
                    W[1, 1, :, :] = Array(bra00 * w, s, s')
                end
            end
        end
        if Nc == 0  #technically only need one boson
            h = Br * tB * (B * onehot(ElT, dag(ri) => r)) + t * Id * D
            w = exp(h) * ket00
            W[1+r, 1, :, :] = Array(bra10 * w, s, s')
            if r == 1
                W[1, 1, :, :] = Array(bra00 * w, s, s')
            end
        end
    end
    if Nr == 0
        for c in eachval(ci)
            h = Bc * tC * (C * onehot(ElT, dag(ci) => c)) + t * Id * D
            w = exp(h) * ket00
            W[1, 1+c, :, :] = Array(bra01 * w, s, s')
            if c == 1
                W[1, 1, :, :] = Array(bra00 * w, s, s')
            end
        end
        if Nc == 0
            W = reshape(Array(exp(t * D), s, s'), 1, 1, d, d)
        end
    end

    W
end


# new Van Damme algorithm for building the MPOs
function makeW(::Algorithm"VD2", ElT, tau, A, B, C, D)
    tC = sqrt(abs(tau))
    tB = tau / tC

    # physical spin indiced (for S=1/2, d=2)
    s = commonind(C, D)
    sp = s'
    ri = noncommonind(A, C)
    ci = noncommonind(A, B)
    
    d = dim(s)
    
    # Recover the true FSM memory dimension (chi) from the expanded link size (chi + chi^2)
    # In opsum_to_U.jl, we forced the virtual link dimension to expand to 1 + chi + chi^2
    # Because the identity track (+1) is stripped out before makeW is called, ri arrives with a dimension of exactly N_expanded = chi + chi^2
    # To find the true FSM memory dimension (chi, which is Nr), we solve the quadratic equation x^2 + x - N_expanded = 0, reaching to x = 0.5 * (-1 + \sqrt{1 + 4N))
    Nr_expanded = isnothing(ri) ? 0 : dim(ri)
    Nc_expanded = isnothing(ci) ? 0 : dim(ci)
    
    Nr = Nr_expanded > 0 ? round(Int, (-1 + sqrt(1 + 4 * Nr_expanded)) / 2) : 0
    Nc = Nc_expanded > 0 ? round(Int, (-1 + sqrt(1 + 4 * Nc_expanded)) / 2) : 0

    # Helper: Convert ITensor to Matrix of physical operator blocks (only up to true chi)
    function to_mat_of_mats(T::ITensor, row_ind, col_ind, true_Nr, true_Nc)
        nrow = isnothing(row_ind) ? 1 : true_Nr
        ncol = isnothing(col_ind) ? 1 : true_Nc
        M = Matrix{Matrix{ElT}}(undef, nrow, ncol)
        for r in 1:nrow
            for c in 1:ncol
                phys_T = T
                if !isnothing(row_ind); phys_T *= onehot(ElT, dag(row_ind) => r); end
                if !isnothing(col_ind); phys_T *= onehot(ElT, dag(col_ind) => c); end
                M[r, c] = Array(phys_T, s, sp)
            end
        end
        return M
    end

    # Helper: Direct Product (Virtual) + Matrix Product (Physical) (No ElT in signature!)
    function otimes(X, Y)
        nx, mx = size(X); ny, my = size(Y)
        Z = Matrix{Matrix{ElT}}(undef, nx * ny, mx * my)
        for ix in 1:nx, jx in 1:mx
            for iy in 1:ny, jy in 1:my
                Z[(ix-1)*ny + iy, (jx-1)*my + jy] = X[ix, jx] * Y[iy, jy]
            end
        end
        return Z
    end

    # Helper: Exact Symmetric Sum 
    function sym_sum(ops...)
        if length(ops) == 1
            return ops[1]
        elseif length(ops) == 2
            A, B = ops
            if A === B
                return otimes(A, A)
            else
                return otimes(A, B) .+ otimes(B, A)
            end
        elseif length(ops) == 3
            A, B, C = ops
            if A === B && B === C
                return otimes(A, otimes(A, A))
            elseif A === B
                return otimes(A, otimes(A, C)) .+ otimes(A, otimes(C, A)) .+ otimes(C, otimes(A, A))
            elseif A === C
                return otimes(A, otimes(B, A)) .+ otimes(A, otimes(A, B)) .+ otimes(B, otimes(A, A))
            elseif B === C
                return otimes(A, otimes(B, B)) .+ otimes(B, otimes(A, B)) .+ otimes(B, otimes(B, A))
            else
                return otimes(A, otimes(B, C)) .+ otimes(A, otimes(C, B)) .+ 
                       otimes(B, otimes(A, C)) .+ otimes(B, otimes(C, A)) .+ 
                       otimes(C, otimes(A, B)) .+ otimes(C, otimes(B, A))
            end
        end
    end

    # Extract Operator Matrices safely within boundaries
    mat_D = tau .* to_mat_of_mats(D, nothing, nothing, Nr, Nc)
    mat_C = Nc > 0 ? (tC .* to_mat_of_mats(C, nothing, ci, Nr, Nc)) : Matrix{Matrix{ElT}}(undef, 1, 0)
    mat_B = Nr > 0 ? (tB .* to_mat_of_mats(B, ri, nothing, Nr, Nc)) : Matrix{Matrix{ElT}}(undef, 0, 1)
    mat_A = (Nr > 0 && Nc > 0) ? to_mat_of_mats(A, ri, ci, Nr, Nc) : Matrix{Matrix{ElT}}(undef, 0, 0)

    I_op = Matrix{Matrix{ElT}}(undef, 1, 1)
    I_op[1, 1] = Matrix{ElT}(I, d, d)

    # Van Damme Appendix A Polynomials 
    W11 = I_op .+ mat_D .+ (1/2) .* sym_sum(mat_D, mat_D) .+ (1/6) .* sym_sum(mat_D, mat_D, mat_D)
    
    W12 = Nc > 0 ? (mat_C .+ (1/2) .* sym_sum(mat_C, mat_D) .+ (1/6) .* sym_sum(mat_C, mat_D, mat_D)) : Matrix{Matrix{ElT}}(undef, 1, 0)
    W13 = Nc > 0 ? (sym_sum(mat_C, mat_C) .+ (1/3) .* sym_sum(mat_C, mat_C, mat_D)) : Matrix{Matrix{ElT}}(undef, 1, 0)

    W21 = Nr > 0 ? (mat_B .+ (1/2) .* sym_sum(mat_B, mat_D) .+ (1/6) .* sym_sum(mat_B, mat_D, mat_D)) : Matrix{Matrix{ElT}}(undef, 0, 1)
    W31 = Nr > 0 ? ((1/2) .* sym_sum(mat_B, mat_B) .+ (1/6) .* sym_sum(mat_B, mat_B, mat_D)) : Matrix{Matrix{ElT}}(undef, 0, 1)

    W22 = (Nr > 0 && Nc > 0) ? (mat_A .+ (1/2) .* (sym_sum(mat_B, mat_C) .+ sym_sum(mat_A, mat_D)) .+ (1/6) .* (sym_sum(mat_C, mat_B, mat_D) .+ sym_sum(mat_A, mat_D, mat_D))) : Matrix{Matrix{ElT}}(undef, 0, 0)
    W23 = (Nr > 0 && Nc > 0) ? (sym_sum(mat_A, mat_C) .+ (1/3) .* (sym_sum(mat_A, mat_C, mat_D) .+ sym_sum(mat_C, mat_C, mat_B))) : Matrix{Matrix{ElT}}(undef, 0, 0)
    W32 = (Nr > 0 && Nc > 0) ? ((1/2) .* sym_sum(mat_A, mat_B) .+ (1/6) .* (sym_sum(mat_A, mat_B, mat_D) .+ sym_sum(mat_B, mat_B, mat_C))) : Matrix{Matrix{ElT}}(undef, 0, 0)
    W33 = (Nr > 0 && Nc > 0) ? (sym_sum(mat_A, mat_A) .+ (1/3) .* (sym_sum(mat_A, mat_B, mat_C) .+ sym_sum(mat_A, mat_A, mat_D))) : Matrix{Matrix{ElT}}(undef, 0, 0)

    # Assemble the Dense ITensor Array
    dim_R = 1 + Nr_expanded
    dim_C = 1 + Nc_expanded
    W_out = zeros(ElT, dim_R, dim_C, d, d)

    function place!(mat, start_row, start_col)
        nr, nc = size(mat)
        for r in 1:nr, c in 1:nc
            W_out[start_row + r - 1, start_col + c - 1, :, :] .= mat[r, c]
        end
    end

    place!(W11, 1, 1)
    if Nc > 0; place!(W12, 1, 2); place!(W13, 1, 2 + Nc); end
    if Nr > 0; place!(W21, 2, 1); place!(W31, 2 + Nr, 1); end
    if Nr > 0 && Nc > 0
        place!(W22, 2, 2);       place!(W23, 2, 2 + Nc)
        place!(W32, 2 + Nr, 2);  place!(W33, 2 + Nr, 2 + Nc)
    end

    return W_out
end