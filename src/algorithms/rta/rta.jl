"""
    compute_in_boundary(P::AbstractArray, A::AbstractArray)

Computes the in-boundary given stochastic matrix ``P`` and truncation set ``A``.
This also works if `P` is a rate matrix.
"""
function compute_in_boundary(P::AbstractArray, Aidx::AbstractArray)
    n  = size(P,1)
    Ac = setdiff(collect(1:n),Aidx)
    return vec(sum(P[Ac,Aidx],dims=1).>0)
end

"""
    rta(
      P::AbstractArray,
      Aidx::AbstractArray;
      saveall::Bool=false)

Repeated Truncation and Augmentation. See, e.g., [1], and references therein.

[1] Kuntz, Juan, et al. "Stationary distributions of continuous-time Markov chains: a review of theory and truncation-based approximations." SIAM Review 63.1 (2021): 3-64.
"""
function rta(
  P::AbstractArray,
  Aidx::AbstractArray;
  saveall::Bool=false)
    ci = compute_in_boundary(P,Aidx)
    ib_A = findall(ci)
    na = size(Aidx,1)
    stat_vec=zeros(size(ib_A,1),na)
    P_A = P[Aidx, Aidx]

    for (i,ibs) in enumerate(ib_A)
        W = copy(P_A)
        W[:,ibs] = W[:,ibs] .+ (1.0 .- sum(W,dims=2))        
        stat_vec[i,:] = get_stat_dist_Q(I-W)
    end

    minvec = minimum(stat_vec, dims=1)
    maxvec = maximum(stat_vec, dims=1)
    avgvec = (minvec+maxvec)/2
    return (minvec=minvec', maxvec=maxvec', avgvec=avgvec')
end

"""
    rta_Q(
      Q::AbstractArray,
      Aidx::AbstractArray;
      saveall=true)

Repeated Truncation and Augmentation for rate matrices.
"""
function rta_Q(
    Q::AbstractArray,
    Aidx::AbstractArray;
    saveall=true)

    Sidx = collect(1:size(Q,1))
    Acidx = setdiff(Sidx, Aidx)
    ci = compute_in_boundary(Q, Aidx)
    ib_A = findall(ci)
    na = size(Aidx,1)
    stat_vec=zeros(size(ib_A,1),na)
    Q_A = Q[Aidx, Aidx]
    out_rates = sum(Q[Aidx, Acidx], dims=2)

    for (i,ibs) in enumerate(ib_A)
        W = copy(Q_A)
        W[:,ibs] = W[:,ibs] .+ out_rates        
        stat_vec[i,:] = get_stat_dist_Q(W)
    end

    minvec = minimum(stat_vec, dims=1)
    maxvec = maximum(stat_vec, dims=1)
    avgvec = (minvec+maxvec)/2
    return (minvec=minvec', maxvec=maxvec', avgvec=avgvec')
end


"""
    rta(
      P::AbstractArray,
      Aidx::AbstractArray,
      rvec::AbstractArray,
      c::Number,
      r::Number)

Repeated Truncation and Augmentation, where bounds on equilibirum expectations
over the entire state space are derivable when a moment boun is given. 

@pre Aidx is an r-sublevel set. 

See [1] for the derivation of the bounds. 
[1] Kuntz, Juan, et al. "Stationary distributions of continuous-time Markov chains: a review of theory and truncation-based approximations." SIAM Review 63.1 (2021): 3-64.
"""
function rta(
  P::AbstractArray,
  Aidx::AbstractArray,
  rvec::AbstractArray,
  c::Number,
  r::Number)
    n = size(P,1)
    minvec, maxvec, avgvec = rta(P, Aidx)

    rta_lbdist = zeros(n)
    rta_lbdist[1:size(Aidx,1)] = minvec*(1-c/r)

    rta_ubdist = zeros(n)
    rta_ubdist[1:size(Aidx,1)] = maxvec

    rta_lb = rta_lbdist'*rvec
    rta_ub = rta_ubdist'*rvec

    return rta_lb, rta_ub
end
