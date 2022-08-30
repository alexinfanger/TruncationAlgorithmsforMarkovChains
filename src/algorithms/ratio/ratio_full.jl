# Dense/full implementation which optimizes for computing
# the "full" stationary distribution, as opposed to just pi(f),
# and also uses dense matrices.

# Notation
# _IO implies the indexing is in the IO framework P11, P12, etc. 
# _K means on K, _A means on A, _S means on S

# List of terms
# pk is stationary distribution of normalized(G)
# piv is the approximation 

# include("../../../functions/LinearAlgebra/Metzler.jl")
"""
    RatioFullParams{T}
Struct containing parameters for ratio algorithm. Currently 
`P12ImP22inv` is stored as a dense matrix, because
Julia does not support sparse solves with sparse right hand sides.
"""
struct RatioFullParams{T}
  P11::T
  P12::T
  P21::T
  P22::T
  P12ImP22inv::Matrix{Float64}
  Ksize::Int64
  Asize::Int64
  Psize::Int64
  Kidx::Vector{Int64}
  AmKidx::Vector{Int64}
end

function RatioFullParams(Kidx::AbstractArray, Aidx::AbstractArray, P::AbstractArray, t)
  if !(issubset(Kidx,Aidx))
    throw(ArgumentError("Kidx not a subset of Aidx"))
  end
  AmKidx = setdiff(Aidx, Kidx)

  P11 = P[Kidx,Kidx]
  P12 = P[Kidx,AmKidx]
  P21 = P[AmKidx,Kidx]
  P22 = P[AmKidx,AmKidx]

  P12ImP22inv = ((I-P22')\(Matrix(P12)'))'

  return RatioFullParams{t}(P11, P12, P21, P22,
                      P12ImP22inv, size(Kidx,1),size(Aidx,1), size(P,1), Kidx, AmKidx)
end

"""
    RatioFullParamsView(Kidx::AbstractArray, Aidx::AbstractArray, P::AbstractArray, t)

Construct `RatioFullParams` using views (to not allocate more space for the 
sub-matrices P11, P12, etc. . This is currently not in use. 
"""
function RatioFullParamsView(Kidx::AbstractArray, Aidx::AbstractArray, P::AbstractArray, t)
  if !(issubset(Kidx,Aidx))
    throw(ArgumentError("Kidx not a subset of Aidx"))
  end
  AmKidx = setdiff(Aidx, Kidx)

  P11 = @view P[Kidx,Kidx]
  P12 = @view P[Kidx,AmKidx]
  P21 = @view P[AmKidx,Kidx]
  P22 = @view P[AmKidx,AmKidx]

  # ImP22inv = I-P22
  P12ImP22inv = ((I-P22')\(Matrix(P12)'))'

  return RatioFullParams{t}(P11, P12, P21, P22,
                      copy(P12ImP22inv), size(Kidx,1),size(Aidx,1), size(P,1), Kidx, AmKidx)
end

"""
    get_template_type(P::AbstractArray)

Returns template type to use for `RatioFullParams` based on P.
"""
function get_template_type(P::AbstractArray)
  Ptype = typeof(P)
  if Ptype in [Tridiagonal{Float64, Vector{Float64}}, SparseMatrixCSC{Float64, Int64}]
    return SparseMatrixCSC{Float64, Int64}
  elseif Ptype == Matrix{Float64}
    return Matrix{Float64}
  else
    throw(ArgumentError("P has unrecognized type. Must be in {Tridiagonal, SparseCSC, Matrix{Float64}."))
  end
end


"""
    orig2IOidx(tp::RatioFullParams, oldvec_S::AbstractVector)

Given vector in _S indexing, return in IO indexing.
"""
function orig2IOidx(tp::RatioFullParams, oldvec_S::AbstractVector)
  newvec_IO = zeros(tp.Asize)
  newvec_IO[1:tp.Ksize] = oldvec_S[tp.Kidx]
  newvec_IO[tp.Ksize+1:end] = oldvec_S[tp.AmKidx]
  return newvec_IO
end


"""
    orig2IOidx(tp::RatioFullParams, oldvec_IO::AbstractVector)

Given vector in _IO indexing, return in _S indexing.
"""
function IO2origidx(tp::RatioFullParams, oldvec_IO::AbstractVector)
  newvec_S = zeros(tp.Psize)
  newvec_S[tp.Kidx] = oldvec_IO[1:tp.Ksize]
  newvec_S[tp.AmKidx] = oldvec_IO[tp.Ksize+1:end]
  return newvec_S
end

# G Renormalization functions

"""
    row_normalize(G::AbstractMatrix; saveall::Bool=false)

Renormalizes an `AbstractMatrix` based on summing the second dimension.
If G is non-negative, this returns a row-stochastic matrix.
"""
function row_normalize(G::AbstractMatrix; saveall::Bool=false)
  Prow = G./sum(G,dims=2)
  return Prow, Dict()
end

"""
    pf_normalize(G::AbstractMatrix; tol=1e-16, saveall::Bool=false)

@pre G should be a non-negative matrix
Renormalize G using the Doob (or Perron-Frobenius) transform.
"""
function pf_normalize(G::AbstractMatrix; tol=1e-16, saveall::Bool=false)
  if any(G.<0)
    throw(ArgumentError("G should be a non-negative matrix."))
  end
  if saveall 
    # P, rho_G, vec_G = DoobTransform_keep_all(G, tol=tol) # currently using Arpack for stability 
    P, rho_G, vec_G = ArpackDoobTransform(G, tol=tol)
    return P, Dict("rho_G"=>rho_G, "vec_G"=> vec_G, "PF_err"=>norm(G*vec_G - rho_G*vec_G , Inf))
  else
    P = DoobTransform(G, tol=tol)
    return P, Dict()
  end
end

"""
    reflect_normalize(G::AbstractMatrix; saveall::Bool=false)
    
@pre local boundary
@pre one-dimensional vector state space
Renormalize G by putting all exiting probability mass into the in-boundary.
Currently assumes it's the one dimensional case.
"""
function reflect_normalize(G::AbstractMatrix; saveall::Bool=false)
  G[end,end] = 1-sum(G[end,:])
  return G, Dict()
end


"""
    get_utilde_kappa(tp::RatioFullParams, r_IO::AbstractArray)

Compute utilde-kappa(``x``), the sum of a reward starting at state ``x`` only 
including paths that return to the set ``K`` before exiting into ``A^c``. This 
is a lower bound (and our approximation to) the sum of rewards starting at state 
``x`` before returning to the set ``K``. 
"""
function get_utilde_kappa(tp::RatioFullParams, r_IO::AbstractArray)
  r1_IO = r_IO[1:tp.Ksize]
  r2_IO = r_IO[(tp.Ksize+1):end]
  return r1_IO .+ tp.P12ImP22inv*r2_IO
end

"""
    pi_K_IO_to_pi_A_IO(
      tp::RatioFullParams, 
      pi_K_approx_IO::AbstractArray;
      e_IO::AbstractArray=ones(tp.Asize)

Given an approximation of ``pi`` conditioned on being in ``K``, returns
the approximation on all of ``A\\supseteq K``.

Currently the optional argument e_IO is not in use. This is because for the 
dense/full implementation of the algorithm, we can just compute the the 
approximation for R and then re-weight it to get the approximation of Q.
"""
function pi_K_IO_to_pi_A_IO(
  tp::RatioFullParams, 
  pi_K_approx_IO::AbstractArray;
  e_IO::AbstractArray=ones(tp.Asize) 
  #                                     
  )
  kappa_e_approx = get_utilde_kappa(tp, e_IO)
  pi_K_dot_kappa_e = pi_K_approx_IO'*kappa_e_approx
  piv = zeros(tp.Asize)

  piv[1:tp.Ksize] = pi_K_approx_IO/pi_K_dot_kappa_e
  piv[(tp.Ksize+1):end] = (pi_K_approx_IO'*tp.P12ImP22inv)/pi_K_dot_kappa_e
  return piv
end

"""
    get_G(tp::RatioFullParams)

Returns ``G = P_{11}+P_{12}(I-P_{22})^{-1}P_{21}``.
"""
function get_G(tp::RatioFullParams)
  return tp.P11 + tp.P12ImP22inv*tp.P21
end

"""
    struct EqExpAndKappaBounds

Struct containing 
      - lb: lower bound on equilibrium expectation
      - ub: upper bounds on equilibrium expectation
      - klr: lower bound on pi_K'kappa(r)
      - kle: lower bound on pi_K'kappa(e)
      - kur: upper bound on pi_K'kappa(r)
      - kue: upper bound on pi_K'kappa(e)
"""
struct EqExpAndKappaBounds
  lb::Float64
  ub::Float64
  klr::Float64
  kle::Float64
  kur::Float64
  kue::Float64
end


"""
    get_CS_r_bounds(
      tp::RatioFullParams, 
      h1_IO::AbstractArray, h2_IO::AbstractArray,
      r_IO::AbstractArray; 
      e_IO::AbstractArray=ones(tp.Asize),
      saveall::Bool=false)

Bounds based on the Courtois and Semal representation. Unstable version.
"""
function get_CS_r_bounds(
  tp::RatioFullParams, 
  h1_IO::AbstractArray, h2_IO::AbstractArray,
  r_IO::AbstractArray; 
  e_IO::AbstractArray=ones(tp.Asize),
  saveall::Bool=false)

  G = sparse(tp.P11 + tp.P12ImP22inv*tp.P21)

  kappa_h1 = get_utilde_kappa(tp, h1_IO)
  kappa_h2 = get_utilde_kappa(tp, h2_IO)

  klr = get_utilde_kappa(tp, r_IO)
  kur = klr + kappa_h1

  kle = get_utilde_kappa(tp, e_IO)
  kue = kle + kappa_h2

  rhs = [klr kur kle kue ones(tp.Ksize)]

  lhs = (I-G)\rhs

  u_klr_IO = lhs[:,1] 
  u_kur_IO = lhs[:,2] 
  u_kle_IO = lhs[:,3]
  u_kue_IO = lhs[:,4]
  u_e_IO = lhs[:,5]


  obj_klr = minimum(u_klr_IO./u_e_IO)
  obj_kur = maximum(u_kur_IO./u_e_IO)
  obj_kle = minimum(u_kle_IO./u_e_IO)
  obj_kue = maximum(u_kue_IO./u_e_IO)

  return EqExpAndKappaBounds(obj_klr/obj_kue, obj_kur/obj_kle, obj_klr, obj_kle, obj_kur, obj_kue)
end

"""
    get_CS_r_bounds_stable(
      tp::RatioFullParams, 
      h1_IO::AbstractArray, h2_IO::AbstractArray,
      r_IO::AbstractArray; 
      e_IO::AbstractArray=ones(tp.Asize),
      saveall::Bool=false)

Bounds based on the Courtois and Semal representation. Stable version.
This implementation requires the user to specify which state z to use when
removing that corresponding row and column from G.
"""
function get_CS_r_bounds_stable(
  tp::RatioFullParams, 
  h1_IO::AbstractArray, h2_IO::AbstractArray,
  r_IO::AbstractArray,
  z_IO::Int64; 
  e_IO::AbstractArray=ones(tp.Asize),
  saveall::Bool=false)

  G = sparse(tp.P11 + tp.P12ImP22inv*tp.P21)

  Gz = G[1:end .!= z_IO, 1:end .!=z_IO]
  pz = G[1:end .!= z_IO, z_IO]

  kappa_h1 = get_utilde_kappa(tp, h1_IO)
  kappa_h2 = get_utilde_kappa(tp, h2_IO)

  klr = get_utilde_kappa(tp, r_IO)
  kur = klr + kappa_h1

  kle = get_utilde_kappa(tp, e_IO)
  kue = kle + kappa_h2

  rhs = [klr[1:end .!= z_IO] kur[1:end .!= z_IO] kle[1:end .!= z_IO] kue[1:end .!= z_IO] ones(tp.Ksize-1) Vector(pz)]
  
  lhs = (I-Gz)\rhs

  prg = G[z_IO, z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,6]

  u_klr_IO = ones(tp.Ksize)*NaN
  u_kur_IO = ones(tp.Ksize)*NaN 
  u_kle_IO = ones(tp.Ksize)*NaN
  u_kue_IO = ones(tp.Ksize)*NaN
  u_e_IO = ones(tp.Ksize)*NaN

  # These are not formally the u-values, but a scaling thereof. We only care 
  # their ratio, so this is fine. See "Implications For Our Paper .pdf".

  u_klr_IO[1:end .!= z_IO] = lhs[:,1]*(1-prg) .+ prg*(klr[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,1])
  u_kur_IO[1:end .!= z_IO] = lhs[:,2]*(1-prg) .+ prg*(kur[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,2]) 
  u_kle_IO[1:end .!= z_IO] = lhs[:,3]*(1-prg) .+ prg*(kle[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,3]) 
  u_kue_IO[1:end .!= z_IO] = lhs[:,4]*(1-prg) .+ prg*(kue[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,4]) 
  u_e_IO[1:end .!= z_IO]   = lhs[:,5]*(1-prg) .+ prg*(1 .+ G[z_IO,1:end .!=z_IO]'*lhs[:,5]) 

  u_klr_IO[z_IO] = klr[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,1]
  u_kur_IO[z_IO] = kur[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,2]
  u_kle_IO[z_IO] = kle[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,3]
  u_kue_IO[z_IO] = kue[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,4]
  u_e_IO[z_IO] =   1.0      .+ G[z_IO,1:end .!=z_IO]'*lhs[:,5]


  obj_klr = minimum(u_klr_IO./u_e_IO)
  obj_kur = maximum(u_kur_IO./u_e_IO)
  obj_kle = minimum(u_kle_IO./u_e_IO)
  obj_kue = maximum(u_kue_IO./u_e_IO)

  return EqExpAndKappaBounds(obj_klr/obj_kue, obj_kur/obj_kle, obj_klr, obj_kle, obj_kur, obj_kue)
end

"""
    get_CS_r_bounds_stable_max_row_sum_heuristic(
      tp::RatioFullParams, 
      h1_IO::AbstractArray, h2_IO::AbstractArray,
      r_IO::AbstractArray;
      e_IO::AbstractArray=ones(tp.Asize),
      saveall::Bool=false)

Bounds based on the Courtois and Semal representation. Stable version.
This implementation deletes the row and column z of G, where z corresponds
to a maximum row sum of G. (The idea being to try to reduce the spectral radius
as much as possible.)
"""
function get_CS_r_bounds_stable_max_row_sum_heuristic(
  tp::RatioFullParams, 
  h1_IO::AbstractArray, h2_IO::AbstractArray,
  r_IO::AbstractArray;
  e_IO::AbstractArray=ones(tp.Asize),
  saveall::Bool=false)

  G = sparse(tp.P11 + tp.P12ImP22inv*tp.P21)
  (_, z_IO) = findmax(G*ones(tp.Ksize))

  Gz = G[1:end .!= z_IO, 1:end .!=z_IO]
  pz = G[1:end .!= z_IO, z_IO]

  kappa_h1 = get_utilde_kappa(tp, h1_IO)
  kappa_h2 = get_utilde_kappa(tp, h2_IO)

  klr = get_utilde_kappa(tp, r_IO)
  kur = klr + kappa_h1

  kle = get_utilde_kappa(tp, e_IO)
  kue = kle + kappa_h2

  rhs = [klr[1:end .!= z_IO] kur[1:end .!= z_IO] kle[1:end .!= z_IO] kue[1:end .!= z_IO] ones(tp.Ksize-1) Vector(pz)]
  
  lhs = (I-Gz)\rhs

  prg = G[z_IO, z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,6]

  u_klr_IO = ones(tp.Ksize)*NaN
  u_kur_IO = ones(tp.Ksize)*NaN 
  u_kle_IO = ones(tp.Ksize)*NaN
  u_kue_IO = ones(tp.Ksize)*NaN
  u_e_IO = ones(tp.Ksize)*NaN

  # These are not formally the u-values, but a scaling thereof. We only care 
  # their ratio, so this is fine. See "Implications For Our Paper .pdf".

  u_klr_IO[1:end .!= z_IO] = lhs[:,1]*(1-prg) .+ prg*(klr[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,1])
  u_kur_IO[1:end .!= z_IO] = lhs[:,2]*(1-prg) .+ prg*(kur[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,2]) 
  u_kle_IO[1:end .!= z_IO] = lhs[:,3]*(1-prg) .+ prg*(kle[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,3]) 
  u_kue_IO[1:end .!= z_IO] = lhs[:,4]*(1-prg) .+ prg*(kue[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,4]) 
  u_e_IO[1:end .!= z_IO]   = lhs[:,5]*(1-prg) .+ prg*(1 .+ G[z_IO,1:end .!=z_IO]'*lhs[:,5]) 

  u_klr_IO[z_IO] = klr[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,1]
  u_kur_IO[z_IO] = kur[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,2]
  u_kle_IO[z_IO] = kle[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,3]
  u_kue_IO[z_IO] = kue[z_IO] + G[z_IO,1:end .!=z_IO]'*lhs[:,4]
  u_e_IO[z_IO] =   1.0      .+ G[z_IO,1:end .!=z_IO]'*lhs[:,5]


  obj_klr = minimum(u_klr_IO./u_e_IO)
  obj_kur = maximum(u_kur_IO./u_e_IO)
  obj_kle = minimum(u_kle_IO./u_e_IO)
  obj_kue = maximum(u_kue_IO./u_e_IO)

  return EqExpAndKappaBounds(obj_klr/obj_kue, obj_kur/obj_kle, obj_klr, obj_kle, obj_kur, obj_kue)
end


"""
     get_pi_3(tp::RatioFullParams)

Computes ``\\pi_3``. This implementation assumes ``S'=A``.
"""
function get_pi_3(tp::RatioFullParams)
  @warn "This implementation currently assumes S' is all of A."

  Pm = [tp.P11 tp.P12; tp.P21 tp.P22]

  e1 = ones(tp.Ksize)
  u2 = (I-tp.P22)\(tp.P21*e1)

  G = tp.P11+tp.P12ImP22inv*tp.P21

  u1 = sum(G, dims=2)
  u = zeros(tp.Asize)
  u[1:tp.Ksize] = u1
  u[tp.Ksize+1:end] = u2

  Du = diagm(u)
  Duinv = diagm(1.0 ./ u)

  R = zeros(tp.Asize,tp.Asize)
  R[:, 1:tp.Ksize] = Duinv*Pm[:, 1:tp.Ksize]
  R[:, tp.Ksize+1:end] = Duinv * Pm[:, tp.Ksize+1:end] * Du[tp.Ksize+1:end, tp.Ksize+1:end]

  pi_3 = get_stat_dist_Q(R-I)

  return pi_3
end

#//////////////////////////////////////////////////////////////////////////////
# Full Algorithms
# Approximations
#//////////////////////////////////////////////////////////////////////////////

struct IOApx_i
  piv_S::Vector{Float64}
  MethodSpecificQuantities::Dict
end

"""
    approx(Kidx::AbstractArray, Aidx::AbstractArray,
                P::AbstractArray, renormalize::Function;
                saveall::Bool=false)

Computes the approximation. 
@pre renormalize is a function that takes in a non-negative matrix and outputs 
  a stochastic matrix. See e.g. row_normalize and pf_normalize above.
"""
function approx(Kidx::AbstractArray, Aidx::AbstractArray,
                P::AbstractArray, renormalize::Function;
                saveall::Bool=false)
  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)
  G = get_G(tp)
  P_i, out_dict = renormalize(G, saveall=saveall)
  pk_IO = get_stat_dist_Q(P_i-I)
  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pk_IO)
  piv_S = IO2origidx(tp, piv_A_IO)
  if saveall
    out_dict["pk_IO"] = pk_IO
    out_dict["piv_A_IO"] = piv_A_IO
  end
  return IOApx_i(piv_S, out_dict)
end

"""
    approx(Kidx::AbstractArray, Aidx::AbstractArray,
                P::AbstractArray, renormalize::Function;
                saveall::Bool=false)

Computes the approximation. Assumes RatioFullParams already computed. 
"""
function approx(
  tp::RatioFullParams,
  renormalize::Function;
  saveall::Bool=false)
  G = get_G(tp)
  P_i, out_dict = renormalize(G, saveall=saveall)
  pk_IO = get_stat_dist_Q(P_i-I)
  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pk_IO)
  piv_S = IO2origidx(tp, piv_A_IO)
  if saveall
    out_dict["pk_IO"] = pk_IO
    out_dict["piv_A_IO"] = piv_A_IO
  end
  return IOApx_i(piv_S, out_dict)
end

"""
   approx_view(Kidx::AbstractArray, Aidx::AbstractArray,
                P::AbstractArray, renormalize::Function;
                saveall::Bool=false)

View version of approx. Currently not being used. 
"""
function approx_view(Kidx::AbstractArray, Aidx::AbstractArray,
                P::AbstractArray, renormalize::Function;
                saveall::Bool=false)
  t1 = typeof(P)
  t = SubArray{Float64, 2, t1, Tuple{Vector{Int64}, Vector{Int64}}, false}
  tp = RatioFullParamsView(Kidx, Aidx, P, t)
  G = get_G(tp)
  P_i, out_dict = renormalize(G, saveall=saveall)
  pk_IO = get_stat_dist_Q(P_i-I)
  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pk_IO)
  piv_S = IO2origidx(tp, piv_A_IO)
  if saveall
    out_dict["pk_IO"] = pk_IO
    out_dict["piv_A_IO"] = piv_A_IO
  end
  return IOApx_i(piv_S, out_dict)
end


"""
    pi3_approx(Kidx::AbstractArray, Aidx::AbstractArray,
               P::AbstractArray; saveall=true)
      
Compute ``\\pi_3``. Currently assumes ``S'=A``.
"""
function pi3_approx(Kidx::AbstractArray, Aidx::AbstractArray,
                     P::AbstractArray; saveall=true)
  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)
  piv_A_IO = get_pi_3(tp)
  piv_S = IO2origidx(tp, piv_A_IO)
  return IOApx_i(piv_S, Dict())
end

function pi3_approx(tp::RatioFullParams; saveall=true)
  piv_A_IO = get_pi_3(tp)
  piv_S = IO2origidx(tp, piv_A_IO)
  return IOApx_i(piv_S, Dict())
end

"""
    conditional_approx(
      Kidx::AbstractArray,
      Aidx::AbstractArray,
      P::AbstractArray, 
      piv_S::AbstractArray;
      saveall::Bool=false)

Given a reference solution on all of _S, run the ratio approximation with the 
correct conditional distribution on K.
"""
function conditional_approx(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray, 
  piv_S::AbstractArray;
  saveall::Bool=false)
    t = get_template_type(P)
    tp = RatioFullParams(Kidx, Aidx, P, t)
    pk_IO = zeros(tp.Ksize)
    pk_IO = piv_S[Kidx]./sum(piv_S[Kidx])
    piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pk_IO)
    piv_S = IO2origidx(tp, piv_A_IO)
    return IOApx_i(piv_S, Dict())
end

function conditional_approx(tp::RatioFullParams,
                            piv_S::AbstractArray;
                            saveall::Bool=false)
      pk_IO = zeros(tp.Ksize)
      pk_IO = piv_S[tp.Kidx]./sum(piv_S[tp.Kidx])
      piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pk_IO)
      piv_S = IO2origidx(tp, piv_A_IO)
      return IOApx_i(piv_S, Dict())
end

"""
    function exit_approx(
      Aidx::AbstractArray,
      P::AbstractArray;
      k::Number=1, 
      saveall::Bool=false)
Exit approximation, also known as truncation and augmentation.
"""
function exit_approx(
  Aidx::AbstractArray,
  P::AbstractArray;
  k::Number=1, 
  saveall::Bool=false)
  
  B = P[Aidx, Aidx]
  hk_A = (I-B')\(e_i(k,size(Aidx,1)))
  if saveall
    sd = Dict("condImB"=>cond(I-B'))
  else
    sd = Dict()
  end
  piv_S = zeros(size(P,1))
  piv_S[Aidx] = hk_A/sum(hk_A)
  return IOApx_i(piv_S, sd)
end

"""
    function exit_approx(
      Aidx::AbstractArray,
      P::AbstractArray;
      k::Number=1, 
      saveall::Bool=false)
Exit approximation, also known as truncation and augmentation.
Input is a rate matrix in this case.
"""
function exit_approx_Q(
  Aidx::AbstractArray,
  Q::AbstractArray; 
  k::Number=1,
  saveall::Bool=false)

  ImBp = (Q[Aidx, Aidx])'
  hk_A = ImBp\(e_i(k,size(Aidx,1)))
  if saveall
    sd = Dict("condImB"=>cond(ImBp))
  else
    sd = Dict()
  end
  piv_S = zeros(size(Q,1))
  piv_S[Aidx] = hk_A/sum(hk_A)
  return IOApx_i(piv_S, sd)
end

"""
CS_ImG_weighted_approx(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray, 
  saveall::Bool=false 
  ) 

This approximation for pi_K uses a weighted combination of the rows of G.
"""
function CS_ImG_weighted_approx(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray, 
  saveall::Bool=false 
  ) 
  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)
  G = get_G(tp)
  ImGinv = (I-Matrix(G))\I(tp.Ksize)
  ImGrsums = ImGinv*ones(tp.Ksize)
  pi_weightedCS = ImGrsums'*ImGinv
  pi_i = pi_weightedCS'/(pi_weightedCS*ones(tp.Ksize))

  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pi_i)
  piv_S = IO2origidx(tp, piv_A_IO)
  if saveall
    sd = Dict("condImB"=>cond(ImBp))
  else
    sd = Dict()
  end
  IOApx_i(piv_S, sd)
end

"""
    CS_min_tv_approx(
      Kidx::AbstractArray,
      Aidx::AbstractArray,
      P::AbstractArray, 
      saveall::Bool=false)

Approximation on approximating pi_K with the two most similar rows of G in 
total variation. 
"""
function CS_min_tv_approx(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray, 
  saveall::Bool=false)

  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)
  G = get_G(tp)
  ImGinv = (I-Matrix(G))\I(tp.Ksize)
  ImGinvrsums = ImGinv*ones(tp.Ksize)
  nImGinv = ImGinv./ImGinvrsums

  ms = mapslices(x->[x], nImGinv, dims=2)[:]
  it = combinations(ms, 2)
  minval, minidx = findmin(map(x->norm(x[1]-x[2],1),it))
  row1, row2 = collect(it)[minidx]
  pi_i = 0.5*row1 + 0.5*row2

  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pi_i)
  piv_S = IO2origidx(tp, piv_A_IO)
  if saveall
    sd = Dict("condImB"=>cond(ImBp))
  else
    sd = Dict()
  end
  IOApx_i(piv_S, sd)
end



#//////////////////////////////////////////////////////////////////////////////
# Full Algorithms
# Bounds
#//////////////////////////////////////////////////////////////////////////////


"""
    CS_r_bounds(
      Kidx::AbstractArray,
      Aidx::AbstractArray,
      P::AbstractArray, 
      r_S::AbstractArray,
      h1_S::AbstractArray,
      h2_S::AbstractArray;
      e_S::AbstractArray=ones(size(h1_S,1)),
      saveall::Bool=false, 
      )

Compute equilibrium expectation bounds based on the Courtois and Semal 
representation.
"""
function CS_r_bounds(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray, 
  r_S::AbstractArray,
  h1_S::AbstractArray,
  h2_S::AbstractArray;
  e_S::AbstractArray=ones(size(h1_S,1)),
  saveall::Bool=false, 
  )
  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)

  h1_A_IO = orig2IOidx(tp, h1_S)
  h2_A_IO = orig2IOidx(tp, h2_S)
  r_A_IO = orig2IOidx(tp, r_S)
  e_A_IO = orig2IOidx(tp, e_S)
  
  return get_CS_r_bounds(tp, h1_A_IO, h2_A_IO, r_A_IO, e_IO=e_A_IO, saveall=saveall)
end

"""
    CS_r_bounds_stable(
      Kidx::AbstractArray,
      Aidx::AbstractArray,
      P::AbstractArray,
      r_S::AbstractArray,
      h1_S::AbstractArray,
      h2_S::AbstractArray,
      z_IO::Int64;
      e_S::AbstractArray=ones(size(h1_S,1)),
      saveall::Bool=false, 
      )

Compute equilibrium expectation bounds based on the Courtois and Semal 
representation. This version removes row and column z_IO in order to 
make computation with I-G more stable.
"""
function CS_r_bounds_stable(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray,
  r_S::AbstractArray,
  h1_S::AbstractArray,
  h2_S::AbstractArray,
  z_IO::Int64;
  e_S::AbstractArray=ones(size(h1_S,1)),
  saveall::Bool=false, 
  )
  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)

  h1_A_IO = orig2IOidx(tp, h1_S)
  h2_A_IO = orig2IOidx(tp, h2_S)
  r_A_IO = orig2IOidx(tp, r_S)
  e_A_IO = orig2IOidx(tp, e_S)
  
  return get_CS_r_bounds_stable(tp, h1_A_IO, h2_A_IO, r_A_IO, z_IO, e_IO=e_A_IO, saveall=saveall)
end


"""
    CS_r_bounds_stable_max_row_sum_heuristic(
      Kidx::AbstractArray,
      Aidx::AbstractArray,
      P::AbstractArray,
      r_S::AbstractArray,
      h1_S::AbstractArray,
      h2_S::AbstractArray;
      e_S::AbstractArray=ones(size(h1_S,1)),
      saveall::Bool=false, 
      )

Compute equilibrium expectation bounds based on the Courtois and Semal 
representation. This version removes row and column z_{max} in order to 
make computation with I-G more stable, where z_{max} corresponds to a maximal
row sum.
"""
function CS_r_bounds_stable_max_row_sum_heuristic(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray,
  r_S::AbstractArray,
  h1_S::AbstractArray,
  h2_S::AbstractArray;
  e_S::AbstractArray=ones(size(h1_S,1)),
  saveall::Bool=false, 
  )
  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)

  h1_A_IO = orig2IOidx(tp, h1_S)
  h2_A_IO = orig2IOidx(tp, h2_S)
  r_A_IO = orig2IOidx(tp, r_S)
  e_A_IO = orig2IOidx(tp, e_S)
  
  return get_CS_r_bounds_stable_max_row_sum_heuristic(tp, h1_A_IO, h2_A_IO, r_A_IO, e_IO=e_A_IO, saveall=saveall)
end

"""
    CS_tv_bounds(
      Kidx::AbstractArray,
      Aidx::AbstractArray,
      P::AbstractArray,
      r_S::AbstractArray,
      h1_S::AbstractArray,
      h2_S::AbstractArray,
      renormalize::Function;
      e_S::AbstractArray=ones(size(h1_S,1)),
      saveall::Bool=false)

Returns Courtois and Semal based tv bounds.
"""
function CS_tv_bounds(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray,
  r_S::AbstractArray,
  h1_S::AbstractArray,
  h2_S::AbstractArray,
  renormalize::Function;
  e_S::AbstractArray=ones(size(h1_S,1)),
  saveall::Bool=false)
  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)
  G = sparse(get_G(tp))
  P_i, out_dict = renormalize(G, saveall=saveall)
  pi_i = get_stat_dist_Q(P_i-I)
  nImG, out_dict_nImG = row_normalize((I-G)\I(tp.Ksize))
  ms = mapslices(x->[x], nImG, dims=2)[:]
  it = combinations(ms, 2)
  max_1_norm = maximum(map(x->norm(x[1]-x[2],1), it))
  Delta_i_bound = max_1_norm

  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pi_i)
  piv_S = IO2origidx(tp, piv_A_IO)
  val = piv_S[Aidx]'*r_S[Aidx]/(piv_S[Aidx]'*e_S[Aidx])

  h1_IO = orig2IOidx(tp, h1_S)
  h2_IO = orig2IOidx(tp, h2_S)
  r_IO = orig2IOidx(tp, r_S)
  r2_IO = orig2IOidx(tp, e_S)

  beta_1 = get_utilde_kappa(tp, h1_IO)
  beta_2 = get_utilde_kappa(tp, h2_IO)

  klr = get_utilde_kappa(tp, r_IO)
  kur = klr + beta_1

  kle = get_utilde_kappa(tp, r2_IO)
  kue = kle + beta_2

  uir = min(maximum(kur),(pi_i'*kur + Delta_i_bound*norm(kur,Inf)))
  uie = min(maximum(kue),(pi_i'*kue + Delta_i_bound*norm(kue,Inf)))

  lir =  max(minimum(klr), (pi_i'*klr - Delta_i_bound*norm(klr,Inf)))
  lie = max(minimum(kle), (pi_i'*kle - Delta_i_bound*norm(kle,Inf)))

  lb = lir/uie
  ub = uir/lie

  if tp.Ksize ==1 
    tvb = 2*max(beta_1[1]/kle[1], val*beta_2[1]/kle[1])
  else
    epsilon_i = ((pi_i'*beta_1 + val*(pi_i'*beta_2))/lie + Delta_i_bound*((val*norm(kue,Inf)+norm(kur,Inf))/lie))
    tvb = 2*epsilon_i
  end

  if saveall
    out_dict["pi"] = pi_i
    out_dict["Pi"] = Pi_i'
    out_dict["rho_G"] = spectral_radius(G)
    return val, IOBds_i(cond(F_i), Delta_i_bound, minimum(sum(I-P_i+Pi_i,dims=2)),
            maximum(sum(I-P_i+Pi_i, dims=2)), lb, ub, tvb, out_dict)
  else
    return BoundApproxOut((val, lb, ub,tvb))
  end
end

"""
  function CS_tv_bounds_ImG(
    Kidx::AbstractArray, 
    Aidx::AbstractArray,
    P::AbstractArray, 
    r_S::AbstractArray,
    h1_S::AbstractArray, 
    h2_S::AbstractArray,
    renormalize::Function;
    e_S::AbstractArray=ones(size(h1_S,1)),
    saveall::Bool=false, 
    )  

Returns CS total variation bounds where the approximation of pi_K is given 
by a weighted combination of the rows of ``(I-G)^{-1}``.
"""
function CS_tv_bounds_ImG(
  Kidx::AbstractArray, 
  Aidx::AbstractArray,
  P::AbstractArray, 
  r_S::AbstractArray,
  h1_S::AbstractArray, 
  h2_S::AbstractArray,
  renormalize::Function;
  e_S::AbstractArray=ones(size(h1_S,1)),
  saveall::Bool=false, 
  ) 
  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)
  G = sparse(get_G(tp))
  P_i, out_dict = renormalize(G, saveall=saveall)
  pi_i = get_stat_dist_Q(P_i-I)
  ImGinv = (I-G)\I(tp.Ksize)
  ImGrsums = ImGinv*ones(tp.Ksize)
  pi_weightedCS = ImGrsums'*ImGinv
  pi_i = pi_weightedCS./(pi_weightedCS'*ones(tp.Ksize))
  nImG, out_dict_nImG = row_normalize(ImGinv)
  ms = mapslices(x->[x], nImG, dims=2)[:]
  it = combinations(ms, 2)
  max_1_norm = maximum(map(x->norm(x[1]-x[2],1), it))
  Delta_i_bound = max_1_norm

  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pi_i)
  piv_S = IO2origidx(tp, piv_A_IO)
  val = piv_S[Aidx]'*r_S[Aidx]/(piv_S[Aidx]'*e_S[Aidx])

  h1_IO = orig2IOidx(tp, h1_S)
  h2_IO = orig2IOidx(tp, h2_S)
  r_IO = orig2IOidx(tp, r_S)
  r2_IO = orig2IOidx(tp, e_S)

  beta_1 = get_utilde_kappa(tp, h1_IO)
  beta_2 = get_utilde_kappa(tp, h2_IO)

  klr = get_utilde_kappa(tp, r_IO)
  kur = klr + beta_1

  kle = get_utilde_kappa(tp, r2_IO)
  kue = kle + beta_2

  uir = min(maximum(kur),(pi_i'*kur + Delta_i_bound*norm(kur,Inf)))
  uie = min(maximum(kue),(pi_i'*kue + Delta_i_bound*norm(kue,Inf)))

  lir =  max(minimum(klr), (pi_i'*klr - Delta_i_bound*norm(klr,Inf)))
  lie = max(minimum(kle), (pi_i'*kle - Delta_i_bound*norm(kle,Inf)))

  lb = lir/uie
  ub = uir/lie

  if tp.Ksize ==1 
    tvb = 2*max(beta_1[1]/kle[1], val*beta_2[1]/kle[1])
  else
    epsilon_i = ((pi_i'*beta_1 + val*(pi_i'*beta_2))/lie + Delta_i_bound*((val*norm(kue,Inf)+norm(kur,Inf))/lie))
    tvb = 2*epsilon_i
  end

  if saveall
    out_dict["pi"] = pi_i
    out_dict["Pi"] = Pi_i'
    out_dict["rho_G"] = spectral_radius(G)
    return val, IOBds_i(cond(F_i), Delta_i_bound, minimum(sum(I-P_i+Pi_i,dims=2)),
            maximum(sum(I-P_i+Pi_i, dims=2)), lb, ub, tvb, out_dict)
  else
    return BoundApproxOut((val, lb, ub,tvb))
  end
end

# Unnecessary redefinition (defined in Markov_process.jl). 
# Leaving for easy acccess when reading this code.
# """
#     NamedTuple BoundApproxOut
# Data structure for keeping
#   - approx: equilibrium expectation approximation
#   - lb: lower bound on equilibrium expectation
#   - ub: upper bound on equilibrium expectation
#   - tv: r-weighted total variation bound
# """
# BoundApproxOut = @NamedTuple begin
#   approx::Float64
#   lb::Float64
#   ub::Float64
#   tvb::Float64
# end

"""
    ptb_bounds(
      Kidx::AbstractArray,
      Aidx::AbstractArray,
      P::AbstractArray,
      r_S::AbstractArray,
      h1_S::AbstractArray,
      h2_S::AbstractArray,
      renormalize::Function;
      saveall::Bool=false)

Pertrubation bounds. 
"""
function ptb_bounds(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray,
  r_S::AbstractArray,
  h1_S::AbstractArray,
  h2_S::AbstractArray,
  renormalize::Function;
  saveall::Bool=false)

  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)
  G = get_G(tp)
  n = sum(G,dims=2)
  delta = minimum(n)
  P_i, out_dict = renormalize(G, saveall=saveall)
  pi_i = get_stat_dist_Q(P_i-I)
  Pi_i = (pi_i*ones(tp.Ksize)')'
  F_i = (I-P_i+Pi_i)\I(tp.Ksize)
  Delta_i_bound = opnorm((P_i-G)*F_i, Inf) + (1-delta)*opnorm(F_i,Inf)

  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pi_i)
  piv_S = IO2origidx(tp, piv_A_IO)
  val = piv_S[Aidx]'*r_S[Aidx]

  h1_IO = orig2IOidx(tp, h1_S)
  h2_IO = orig2IOidx(tp, h2_S)
  r_IO = orig2IOidx(tp, r_S)
  r2_IO = orig2IOidx(tp, ones(tp.Asize))

  beta_1 = get_utilde_kappa(tp, h1_IO)
  beta_2 = get_utilde_kappa(tp, h2_IO)

  klr = get_utilde_kappa(tp, r_IO)
  kur = klr + beta_1

  kle = get_utilde_kappa(tp, r2_IO)
  kue = kle + beta_2

  uir = min(maximum(kur),(pi_i'*kur + Delta_i_bound*norm(kur,Inf)))
  uie = min(maximum(kue),(pi_i'*kue + Delta_i_bound*norm(kue,Inf)))

  lir =  max(minimum(klr), (pi_i'*klr - Delta_i_bound*norm(klr,Inf)))
  lie = max(minimum(kle), (pi_i'*kle - Delta_i_bound*norm(kle,Inf)))

  lb = lir/uie
  ub = uir/lie

  if tp.Ksize ==1 
    tvb = 2*max(beta_1[1]/kle[1], val*beta_2[1]/kle[1])
  else
    epsilon_i = ((pi_i'*beta_1 + val*(pi_i'*beta_2))/lie + Delta_i_bound*((val*norm(kue,Inf)+norm(kur,Inf))/lie))
    tvb = 2*epsilon_i
  end

  if saveall
    out_dict["pi"] = pi_i
    out_dict["Pi"] = Pi_i'
    out_dict["F_i"] = F_i
    out_dict["min_row_sum_G"] = delta
    out_dict["rho_G"] = spectral_radius(G)
    return val, IOBds_i(cond(F_i), Delta_i_bound, minimum(sum(I-P_i+Pi_i,dims=2)),
            maximum(sum(I-P_i+Pi_i, dims=2)), lb, ub, tvb, out_dict)
  else
    return BoundApproxOut((val, lb, ub,tvb))
  end
end

"""
 ptb_bounds(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray,
  r_S::AbstractArray,
  h1_S::AbstractArray,
  r2_S::AbstractArray,
  h2_S::AbstractArray,
  renormalize::Function;
  saveall::Bool=false)


Perturbation bounds. Adding the argument `r2_S` is useful for the continuous-time 
case. `r2_S`` is the rescaled `e_S`: ``r2_S(x)= 1/\\lambda(x)``. 
"""
function ptb_bounds(
  Kidx::AbstractArray,
  Aidx::AbstractArray,
  P::AbstractArray,
  r_S::AbstractArray,
  h1_S::AbstractArray,
  r2_S::AbstractArray,
  h2_S::AbstractArray,
  renormalize::Function;
  saveall::Bool=false)

  t = get_template_type(P)
  tp = RatioFullParams(Kidx, Aidx, P, t)
  G = get_G(tp)
  n = sum(G,dims=2)
  delta = minimum(n)
  P_i, out_dict = renormalize(G, saveall=saveall)
  pi_i = get_stat_dist_Q(P_i-I)
  Pi_i = (pi_i*ones(tp.Ksize)')'
  F_i = (I-P_i+Pi_i)\I(tp.Ksize)
  Delta_i_bound = opnorm((P_i-G)*F_i, Inf) + (1-delta)*opnorm(F_i,Inf)

  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pi_i)
  piv_S = IO2origidx(tp, piv_A_IO)
  val = piv_S[Aidx]'*r_S[Aidx]/(piv_S[Aidx]'*r2_S[Aidx])

  h1_IO = orig2IOidx(tp, h1_S)
  h2_IO = orig2IOidx(tp, h2_S)
  r_IO = orig2IOidx(tp, r_S)
  r2_IO = orig2IOidx(tp, r2_S)

  beta_1 = get_utilde_kappa(tp, h1_IO)
  beta_2 = get_utilde_kappa(tp, h2_IO)

  klr = get_utilde_kappa(tp, r_IO)
  kur = klr + beta_1

  kle = get_utilde_kappa(tp, r2_IO)
  kue = kle + beta_2

  uir = min(maximum(kur),(pi_i'*kur + Delta_i_bound*norm(kur,Inf)))
  uie = min(maximum(kue),(pi_i'*kue + Delta_i_bound*norm(kue,Inf)))

  lir =  max(minimum(klr), (pi_i'*klr - Delta_i_bound*norm(klr,Inf)))
  lie = max(minimum(kle), (pi_i'*kle - Delta_i_bound*norm(kle,Inf)))

  lb = lir/uie
  ub = uir/lie

  if tp.Ksize ==1 
    tvb = 2*max(beta_1[1]/kle[1], val*beta_2[1]/kle[1])
  else
    epsilon_i = ((pi_i'*beta_1 + val*(pi_i'*beta_2))/lie + Delta_i_bound*((val*norm(kue,Inf)+norm(kur,Inf))/lie))
    tvb = 2*epsilon_i
  end

  if saveall
    out_dict["pi"] = pi_i
    out_dict["Pi"] = Pi_i'
    out_dict["F_i"] = F_i
    out_dict["min_row_sum_G"] = delta
    out_dict["rho_G"] = spectral_radius(G)
    return val, IOBds_i(cond(F_i), Delta_i_bound, minimum(sum(I-P_i+Pi_i,dims=2)),
            maximum(sum(I-P_i+Pi_i, dims=2)), lb, ub, tvb, out_dict)
  else
    return BoundApproxOut((val, lb, ub,tvb))
  end
end

"""
    ptb_bounds(
        tp::RatioFullParams,
        Aidx::AbstractArray,
        r_S::AbstractArray,
        h1_S::AbstractArray,
        h2_S::AbstractArray,
        renormalize::Function; 
        saveall::Bool=false)

Perturbation bounds. Discrete time. Assume RatioFullParams has been computed.
"""
function ptb_bounds(
    tp::RatioFullParams,
    Aidx::AbstractArray,
    r_S::AbstractArray,
    h1_S::AbstractArray,
    h2_S::AbstractArray,
    renormalize::Function; 
    saveall::Bool=false)

  G = get_G(tp)
  n = sum(G,dims=2)
  delta = minimum(n)
  P_i, out_dict = renormalize(G, saveall=saveall)
  pi_i = get_stat_dist_Q(P_i-I)
  Pi_i = (pi_i*ones(tp.Ksize)')'
  F_i = (I-P_i+Pi_i)\I(tp.Ksize)
  Delta_i_bound = opnorm((P_i-G)*F_i, Inf) + (1-delta)*opnorm(F_i,Inf)

  piv_A_IO = pi_K_IO_to_pi_A_IO(tp, pi_i)
  piv_S = IO2origidx(tp, piv_A_IO)
  val = piv_S[Aidx]'*r_S[Aidx]

  h1_IO = orig2IOidx(tp, h1_S)
  h2_IO = orig2IOidx(tp, h2_S)
  r1_IO = orig2IOidx(tp, r_S)
  e_IO = ones(tp.Asize)

  beta_1 = get_utilde_kappa(tp, h1_IO)
  beta_2 = get_utilde_kappa(tp, h2_IO)

  klr = get_utilde_kappa(tp, r1_IO)
  kur = klr + beta_1

  kle = get_utilde_kappa(tp, e_IO)
  kue = kle + beta_2

  uir = min(maximum(kur),(pi_i'*kur + Delta_i_bound*norm(kur,Inf)))
  uie = min(maximum(kue),(pi_i'*kue + Delta_i_bound*norm(kue,Inf)))

  lir =  max(minimum(klr), (pi_i'*klr - Delta_i_bound*norm(klr,Inf)))
  lie = max(minimum(kle), (pi_i'*kle - Delta_i_bound*norm(kle,Inf)))

  lb = lir/uie
  ub = uir/lie

  if tp.Ksize ==1 
    tvb = 2*max(beta_1[1]/kle[1], val*beta_2[1]/kle[1])
  else
    epsilon_i = ((pi_i'*beta_1 + val*(pi_i'*beta_2))/lie + Delta_i_bound*((val*norm(kue,Inf)+norm(kur,Inf))/lie))
    tvb = 2*epsilon_i
  end

  if saveall
    out_dict["pi"] = pi_i
    out_dict["Pi"] = Pi_i'
    out_dict["F_i"] = F_i
    out_dict["min_row_sum_G"] = delta
    out_dict["rho_G"] = spectral_radius(G)
    return val, IOBds_i(cond(F_i), Delta_i_bound, minimum(sum(I-P_i+Pi_i,dims=2)),
            maximum(sum(I-P_i+Pi_i, dims=2)), lb, ub, tvb, out_dict)
  else
    return val, lb, ub, tvb
  end
end


"""
    get_all_condition_numbers_Gz(G::AbstractArray)

Computes the condition number of ``G[-z,-z]`` for all ``z\\in A``. 
"""
function get_all_condition_numbers_Gz(G::AbstractArray)
  ksize = size(G,1)
  conds = zeros(ksize)*NaN
  for i = 1:ksize
    Gz = G[1:end .!= i, 1:end .!= i]
    conds[i] = cond(Matrix(Gz))
  end
  return conds
end
