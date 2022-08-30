"""
    uniformize(Q::AbstractMatrix)

Uniformizes Q to construct a stochatic matrix. Returns the stochastic matrix
``I+\\frac{1}{\\lambda}Q`` and ``\\lambda=\\max_{x}|Q(x,x)|``.
"""
function uniformize(Q::AbstractMatrix)
    n = size(Q)[1]
    dg = diag(Q)
    λ = maximum(abs.(dg))
    return I(n) .+ (1/λ).*Q, λ
end

"""
    uniformize(Q::Tridiagonal)

Uniformizes Q to construct a stochatic matrix. Returns the stochastic matrix
``I+\\frac{1}{\\lambda}Q`` and ``\\lambda=\\max_{x}|Q(x,x)|``.
"""
function uniformize(Q::Tridiagonal)
    P = copy(Q)
    λ = maximum(abs.(P.d))
    lmul!((1/λ), P)
    P.d .+= 1
    return P, λ
end


"""
    get_R_matrix(Q::SparseMatrixCSC{Float64, Int64})

Constructs a stochastic matrix via ``P = D^{-1}(Q+D)`` where ``D=\\text{diag}(|Q|)``.
Returns ``P`` and ``\\text{diag}(|Q|)``.
"""
function get_R_matrix(Q::SparseMatrixCSC{Float64, Int64})
    d = Vector(abs.(diag(Q)))
    D = spdiagm(d)
    Dinv = spdiagm(1.0 ./d)
    return Dinv*(Q+D),d
end


"""
    get_R_matrix(Q::Tridiagonal{Float64, Vector{Float64}})

Constructs a stochastic matrix via ``P = D^{-1}(Q+D)`` where ``D=\\text{diag}(|Q|)``.
"""
function get_R_matrix(Q::Tridiagonal{Float64, Vector{Float64}})
    d = Vector(abs.(diag(Q)))
    D = spdiagm(d)
    Dinv = spdiagm(1.0 ./d)
    return Dinv*(Q+D),d
end


"""
    tv(d1::AbstractArray,d2::AbstractArray)

Compute total variation distance.
"""
function tv(d1::AbstractArray,d2::AbstractArray)
  return .5*norm(d1-d2,1)
end


"""
    tvzp(d1::AbstractArray, d2::AbstractArray)

Compute total variation distance when `d1` and `d2` are vectors that may not 
have the same size. In this case it is assumed that the smaller vector is meant 
to be padded with zeros.
"""
function tvzp(d1::AbstractVector, d2::AbstractVector)
    nd1 = size(d1,1)
    nd2 = size(d2,1)
    if nd1<=nd2
        return tv(d1,d2[1:nd1])+0.5*(sum(d2[(nd1+1):end]))
    else
        return tv(d1[1:nd2],d2)+0.5*(sum(d1[(nd2+1):end]))
    end
end

"""
    tv_restricted(
      d1::AbstractArray, 
      d2::AbstractArray,
      Aidx::AbstractArray)

Compute total variation distance restricts to a subset `Aidx`.
"""
function tv_restricted(
  d1::AbstractVector, 
  d2::AbstractVector,
  Aidx::Vector{Int64})
  return .5*norm(d1[Aidx]-d2[Aidx], 1)
end


function is_rate_matrix(
  Q::Matrix{Float64};
  tol=1e-7)
  is_rate = true
  if any(abs.(sum(Q,dims=2)).>tol)
    is_rate = false
  elseif any((Q-diagm(diag(Q))).<0)
    is_rate = false
  elseif any(diag(Q).>0)
    is_rate = false
  elseif any(isnan.(Q))
    is_rate = false
  elseif any(isinf.(Q))
    is_rate = false
  end
  return is_rate
end


function is_rate_matrix(
  Q::SparseMatrixCSC{Float64, Int64}; 
  tol::Float64=1e-7)
  e_vec = ones(size(Q,1))
  d_vec = diag(Q)
  is_rate =  true    
  if any(abs.(Q*e_vec.>tol))
    is_rate = false
  elseif any((Q-spdiagm(d_vec)).nzval.<0)
    is_rate = false
  elseif any(d_vec.>0)
    is_rate = false
  elseif any(isnan.(Q.nzval))
    is_rate = false
  elseif any(isinf.(Q.nzval))
    is_rate = false
  end
  return is_rate
end

function is_stochastic_matrix(P::Matrix{Float64})
    is_stochastic = true
    if any(abs.(sum(P,dims=2).-1).>1e-10)
        is_stochastic = false
    elseif any(P.<0)
        is_stochastic = false
    elseif any(isnan.(P))
        is_stochastic = false
    elseif any(isinf.(P))
        is_stochastic = false
    end
    return is_stochastic
end

function is_stochastic_matrix(
    P::SparseMatrixCSC{Float64, Int64}; 
    tol::Float64=1e-10)
    e = ones(P.m)
    is_stochastic = true
    if any(abs.(P*e .- 1.0) .> tol)
        is_stochastic = false
    elseif any(P.nzval.<0)
        is_stochastic = false
    elseif any(isnan.(P.nzval))
        is_stochastic = false
    elseif any(isinf.(P.nzval))
        is_stochastic = false
    end
    return is_stochastic
end

function is_stochastic_matrix(
    P::Tridiagonal{Float64, Vector{Float64}}; 
    tol::Float64=1e-10)
    e = ones(P.m)
    is_stochastic = true
    if any(abs.(P*e .- 1.0) .> tol)
        is_stochastic = false
    elseif any(P.nzval.<0)
        is_stochastic = false
    elseif any(isnan.(P.nzval))
        is_stochastic = false
    elseif any(isinf.(P.nzval))
        is_stochastic = false
    end
    return is_stochastic
end

function is_substochastic_matrix(P::Matrix{Float64})
    is_substochastic = true
    if any(sum(P,dims=2).>1)
        is_substochastic = false
    elseif any(P.<0)
        is_substochastic = false
    elseif any(isnan.(P))
        is_substochastic = false
    elseif any(isinf.(P))
        is_substochastic = false
    end
    return is_substochastic
end

function is_substochastic_matrix(
    P::SparseMatrixCSC{Float64, Int64}; 
    tol::Float64=1e-10)
    e = ones(P.m)
    is_substochastic = true
    if any(P*e .- 1.0 .> tol)
        is_substochastic = false
    elseif any(P.nzval.<0)
        is_substochastic = false
    elseif any(isnan.(P.nzval))
        is_substochastic = false
    elseif any(isinf.(P.nzval))
        is_substochastic = false
    end
    return is_substochastic
end

function is_stochastic_vector(piv::AbstractVector)
    is_stochastic = true
    if abs(sum(piv)-1)>1e-8
        is_stochastic = false
    elseif ~all(piv.>=0)
        is_stochastic = false
    elseif any(isnan.(piv))
        is_stochastic = false
    elseif any(isinf.(piv))
        is_stochastic = false
    end
    return is_stochastic
end

function min_row_sum(G::Matrix{Float64})
    return minimum(sum(G,dims=2))
end


"""
    e_i(i::Int64,n::Int64)
@pre n>i
Returns unit basis vector ``e_i``, also known as a one-hot vector.
"""
function e_i(i::Int64,n::Int64)
    vec = zeros(n)
    vec[i]=1
    return vec
end


"""
    rescale_piv(piv::Vector{Float64}, weights::Vector{Float64})

Rescale a stochastic vector `piv`` (short for pi-vector) using `weights`.
"""
function rescale_piv(piv::Vector{Float64}, weights::Vector{Float64})
  a = piv.*weights
  return a/sum(a)
end

"""
    unif_param(Q::AbstractMatrix)

Returns uniformization paramter ``\\max_x |Q(x,x)|``.
"""
function unif_param(Q::AbstractMatrix)
    return maximum(abs.(diag(Q)))
end

"""
    get_stat_dist_Q(Q::AbstractMatrix)

Computes the stationary distribution of an irreducible rate matrix ``Q``.
To compute the stationary distribution of a stochastic matrix ``P``, use
`get_start_dist_Q(P-I)`. This uses the "Remove an Equation" approach of 
Section 2.3.1.3 in [1].

[1] Stewart, William J. Introduction to the numerical solution of Markov chains.
    Princeton University Press, 1994.
"""
function get_stat_dist_Q(Q::AbstractMatrix)
    n = size(Q)[1]
    Qt = Q'
    B = Qt[1:n-1, 1:n-1]
    d = Array{Float64,1}(Qt[1:n-1,n])
    piv = B \ (-1*d)
    piv = vcat(piv,1)
    piv = piv/sum(piv)
    return piv
end
