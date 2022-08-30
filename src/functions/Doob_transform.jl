function ArpackDoobTransform(A::AbstractMatrix)
  pf_val, pf_vec = eigs(A;nev=1)
  pf_val_r = real(pf_val[1])
  pf_vec_r = vec(real(pf_vec))
  D = spdiagm(pf_vec_r)
  Dinv = spdiagm(1.0 ./ pf_vec_r)
  P = Dinv*A*D/pf_val_r
  return P, pf_val_r, pf_vec_r
end

# The below code has not been adequately tested and is not used in the current
# implementation of our algorithm. This is specialized code for finding the 
# pf eigenvector of a nonnegative matrix. Since on our models the PF transform
# did not prove to be a particularly strong approximation, we decided against
# spending time perfecting this algorithm, since row normalization should be faster
# than solving (even a very specialized) eigenvalue problem. 
# This is an implementation of Noda's algorithm (see [1]).
# 
# [1] Takashi Noda. Note on the computation of the maximal eigenvalue of a non-negative
#     irreducible matrix. Numer. Math, 17:382–386, 1971.
# 

function pf_inverse_iteration(N::AbstractArray; tol::Float64=1e-16, x0=nothing, l=nothing, maxiter=1000)
  # Potentially code up a better first guess.
  n = size(N,1)
  if x0 == nothing
    x0 = ones(n)
  end

  if l == nothing
    l = maximum(sum(N,dims=2))
  end

  xp = copy(x0)
  xstar = copy(x0)
  v = ones(n)

  iter =0
  err = 1
  lu = l
  ll = l-1
  while err > tol
    iter +=1
    if iter >= maxiter
      ErrorException("Maxiter.")
      # println("maxiter")
      break;
    end
    # putting this in under the assumption that if inverse iteration gets singular you are done.
    try
      xstar = (lu*I-N)\x0
    catch e
      if typeof(e) == SingularException
        return l, xp, lu, ll
      else
        ErrorException("Something went wrong.")
      end
    end
    tau = (xstar'v)/(x0'v)
    xp = xstar/tau
    tauu = maximum(xstar./x0)
    lu = lu - (1/tauu)
    l = lu-(1/tau)
    taul = minimum(xstar./x0)
    ll = lu - (1/taul)
    err = abs(lu-ll)
    x0 = xp
  end
  return l, xp, lu, ll
end

function DoobTransform(A::AbstractArray; tol=1e-16)
  ld, d, _, _ = pf_inverse_iteration(A, tol=tol)
  D = diagm(d)
  Dinv = diagm(1.0 ./ d)
  P = Dinv*A*D/ld
  return P
end

function DoobTransform_keep_ld(A::AbstractArray; tol=1e-16)
  ld, d, _, _ = pf_inverse_iteration(A, tol=tol)
  D = diagm(d)
  Dinv = diagm(1.0 ./ d)
  return Dinv*A*D/ld, ld
end

function DoobTransform_keep_all(A::AbstractArray; tol=1e-16)
  ld, d, _, _ = pf_inverse_iteration(A, tol=tol)
  D = diagm(d)
  Dinv = diagm(1.0 ./ d)
  P = Dinv*A*D/ld
  return P, ld, d
end
