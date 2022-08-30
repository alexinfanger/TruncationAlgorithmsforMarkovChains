"""
    LyapunovBound2MomentBound(
      P::AbstractArray, 
      g::AbstractArray, 
      r::AbstractArray)

@pre Assumes ``Pg\\leq g-r`` outside of the indices given. 
Given a Lyapunov function, returns a moment bound. 
"""
function LyapunovBound2MomentBound(
  P::AbstractArray, 
  g::AbstractArray, 
  r::AbstractArray)

  c = findmax(P*g-g+r)
  return c[1]
end

"""
    LyapunovBound2MomentBoundQ(
      Q::AbstractArray, 
      g::AbstractArray, 
      r::AbstractArray)

@pre Assumes ``Qg\\leq r`` outside of the indices given. 
Given a Lyapunov function, returns a moment bound. 
"""
function LyapunovBound2MomentBoundQ(
  Q::AbstractArray,
  g::AbstractArray,
  r::AbstractArray)

  c = findmax(Q*g+r)
  return c[1]
end


"""
    LyapunovBound2MomentBoundQ_on_subset(
      Q::AbstractArray, 
      g::AbstractArray, 
      r::AbstractArray,
      Aidx::Vector{Int64})

@pre Assumes ``Qg\\leq r`` outside of ``Aidx``. 
@pre Assumes it is enough to do the matrix multiply ``Q[Aidx, Aidx]*g[Aidx]``.
Given a Lyapunov function, returns a moment bound. 
"""
function LyapunovBound2MomentBoundQ_on_subset(
  Q::AbstractArray,
  g::AbstractArray,
  r::AbstractArray,
  Aidx::Vector{Int64})

  c = findmax(Q[Aidx,Aidx]*g[Aidx]+r[Aidx])
  return c[1]
end

"""
  optimal_h(
    mc::MarkovChain,
    gvec::AbstractArray,
    Acidx::AbstractArray)

Returns the optimal h, namely ``h(x)=\\sum_{y\\in A^c} P(x,y)g(y)``.
"""
function optimal_h(
  mc::MarkovChain,
  gvec::AbstractArray,
  Acidx::AbstractArray)

  hvec = (mc.P)[:,Acidx] * gvec[Acidx]
  return hvec
end


"""
  optimal_h(
    P::AbstractArray,
    gvec::AbstractArray,
    Acidx::AbstractArray)

Returns the optimal h, namely ``h(x)=\\sum_{y\\in A^c} P(x,y)g(y)``.
"""
function optimal_h(
  P::AbstractArray,
  gvec::AbstractArray,
  Acidx::AbstractArray)

  hvec = P[:,Acidx] * gvec[Acidx]
  return hvec
end


"""
    is_Lyapunov_function(
      P::AbstractArray,
      Kc::Vector{Int64},
      r::AbstractArray,
      g::AbstractArray
      )

Checks if ``\\sum_{y\\in K^c} P(x,y)g(y)\\leq g(x)-r(x)`` for ``x\\in K^c``.
"""
function is_Lyapunov_function(
  P::AbstractArray,
  Kc::AbstractArray, 
  r_S::AbstractArray,
  g_S::AbstractArray)
  if all((P[Kc,Kc]*g_S[Kc]).<=g_S[Kc]-r_S[Kc])
    return true
  else
    return false
  end
end


"""
    is_Lyapunov_function_rate(
      Q::AbstractArray,
      Kc::Vector{Int64},
      r::AbstractArray,
      g::AbstractArray
      )

Checks if ``\\sum_{y\\in K^c} Q(x,y)g(y)\\leq -r(x)`` for ``x\\in K^c``.
"""
function is_Lyapunov_function_rate(
  Q::AbstractArray,
  Kc::Vector{Int64},
  r::AbstractArray,
  g::AbstractArray
  )
  if all(Q[Kc,Kc]*g[Kc].<=-r[Kc])
    return true
  else 
    return false
  end
end

"""
    get_K_rate_matrix_simple(
      Q::AbstractArray,
      r_S::AbstractArray,
      f_S::AbstractArray
    )

Computes ``K=\\{x: (Qf)(x)+r>0\\}``. Returns `K, KC, Kbool, v` where 
`KC` is ``K^c``, `v=Qf+r_s` and `Kbool=v.>0``.
"""
function get_K_rate_matrix_simple(
  Q::AbstractArray,
  r_S::AbstractArray,
  f_S::AbstractArray
)
  v = Q*f_S+r_S
  Kbool = v.>0
  K = findall(Kbool)
  KC = setdiff(1:vss.size, K)
  return (K=K, KC=KC, Kbool=Kbool, v=v)
end

"""
    get_K_rate_matrix_simple_on_subset(
      Q::AbstractArray,
      r_S::AbstractArray,
      f_S::AbstractArray,
      Aidx::Vector{Int64}
    )

Computes ``K=\\{x: (Qf)(x)+r>0\\}``. It is assumed that you only need to 
consider ``Q, r_s, f_S`` on a subset ``Aidx`` for computing ``K``.
"""
function get_K_rate_matrix_simple_on_subset(
  Q::AbstractMatrix,
  r_S::AbstractVector,
  f_S::AbstractVector,
  Aidx::Vector{Int64}
)
  Q_A = Q[Aidx, Aidx]
  r_A = r_S[Aidx]
  f_A = f_S[Aidx]
  v_A = Q_A*f_A+r_A
  Kbool_A = v_A.>0
  K = Aidx[Kbool_A]
  return K
end


"""
    get_K_union(
      Q::AbstractArray,
      r1_S::Vector{Float64},
      g1_S::Vector{Float64},
      r2_S::Vector{Float64},
      g2_S::Vector{Float64}
      )

Returns the union of `K`'s computed via `get_K_rate_matrix_simple`.
"""
function get_K_union(
  Q::AbstractArray,
  r1_S::Vector{Float64},
  g1_S::Vector{Float64},
  r2_S::Vector{Float64},
  g2_S::Vector{Float64}
  )
  lo_1 = get_K_rate_matrix_simple(Q, r1_S, g1_S)
  lo_2 = get_K_rate_matrix_simple(Q, r2_S, g2_S)
  return union(lo_1.K, lo_2.K)
end

"""
    get_K_union_on_subset(
      Q::AbstractArray,
      r1_S::Vector{Float64},
      g1_S::Vector{Float64},
      r2_S::Vector{Float64},
      g2_S::Vector{Float64},
      Aidx::Vector{Int64}
      )

Returns the union of `K`'s computed via `get_K_rate_matrix_simple_subset`.
"""
function get_K_union_on_subset(
  Q::AbstractArray,
  r1_S::Vector{Float64},
  g1_S::Vector{Float64},
  r2_S::Vector{Float64},
  g2_S::Vector{Float64},
  Aidx::Vector{Int64}
  )
  K_1 = get_K_rate_matrix_simple_on_subset(Q, r1_S, g1_S, Aidx)
  K_2 = get_K_rate_matrix_simple_on_subset(Q, r2_S, g2_S, Aidx)
  return union(K_1, K_2)
end



"""
    optimal_g(
      P::AbstractArray, 
      g_S::AbstractArray,
      r_S::AbstractArray, 
      Kcidx::AbstractArray)


Given ``P,g,r,K^c``, compute the optimal rescaling of ``g`` so that 
``Pg\\leq g-r`` still holds on ``K^c``. 
"""
function optimal_g(
  P::AbstractArray, 
  g_S::AbstractArray,
  r_S::AbstractArray, 
  Kcidx::AbstractArray)

  v = g_S[Kcidx]-P[Kcidx,Kcidx]*g_S[Kcidx]
  alpha = maximum(r_S[Kcidx]./v)
  return (gmPg=v,alpha=alpha)
end



###################################d
# The code below is used for tests that vary both K and A, 
# which is currently not in the paper. 
###################################

# Code for varying K as sublevel sets. 
# trange version
struct LyapunovMultiKOut
  minKidx::AbstractArray
  minKcidx::AbstractArray
  f1_S::AbstractArray
  f2_S::AbstractArray
  r1_S::AbstractArray
  r2_S::AbstractArray
  c1vec::AbstractArray
  c2vec::AbstractArray
  tmin::Number
  tminidx::Integer
  trange::AbstractArray
  momentbounds::AbstractArray
end

function to_table(lo::LyapunovMultiKOut, rstr::String)
    fstr = ""
    fstr = string(fstr, @sprintf("\\begin{table}\n"))
    fstr = string(fstr, @sprintf("\\begin{tabular}{|l|l|}\n"))
    fstr = string(fstr, @sprintf("\\hline \n"))
    fstr = string(fstr, @sprintf(" Reward Function & %s \\\\ \n", rstr))
    fstr = string(fstr, @sprintf(" \$K_{\\min}\$  & \$\\{x: e^Tx<%d \\}\$ \\\\ \n", lo.tmin))
    fstr = string(fstr,@sprintf("\\hline\n"))
    fstr = string(fstr,@sprintf("\\end{tabular}\n"))
    fstr = string(fstr,@sprintf("\\end{table}\n"))
    print(fstr)
    return
end

function get_momentbounds(P::AbstractArray, 
                          r1_S::AbstractArray,
                          f1_S::AbstractArray,
                          c1vec::AbstractArray,
                          trange::AbstractArray)
  nb = size(c1vec,1)                          
  momentbounds = zeros(nb)
  for i=1:nb
    momentbounds[i] = LyapunovBound2MomentBound(P, c1vec[i]*f1_S,r1_S)
  end
  return momentbounds
end

function get_all_K(mc::MarkovChain,
                   r1_S::AbstractArray, f1_S::AbstractArray,
                   r2_S::AbstractArray, f2_S::AbstractArray,
                   trange::AbstractArray, w::Function; 
                   tol=1e-12)

  P = mc.P
  n = size(P,1)

  boolvec = (P*f1_S.<f1_S) .& (P*f2_S.<f2_S)
  if all(boolvec.==0)
    throw(ArgumentError("No K available for this Lyapunov function."))
  end

  c1vec = []
  c2vec = []
  tmin = 0
  flag = false
  minKidx = []
  minKcidx = []
  tminidx = 0

  Aidx = collect(1:n)

  for (i, ti) in enumerate(trange)
    Kidx = get_sublevelset(mc.vss, w, ti)
    if !(issubset(Kidx, Aidx))
      break
    end
    Kcidx = setdiff(Aidx, Kidx)
    if (any((P[Kcidx,Kcidx]*f1_S[Kcidx])>=f1_S[Kcidx]) ||
        any((P[Kcidx,Kcidx]*f2_S[Kcidx])>=f2_S[Kcidx]))
        continue
    elseif flag==false
      tmin = ti
      tminidx = i
      minKidx = Kidx
      minKcidx = Kcidx
      flag = true
    end

    w1vec = (f1_S[Kcidx] - P[Kcidx, Kcidx]*f1_S[Kcidx])
    w2vec = (f2_S[Kcidx] - P[Kcidx, Kcidx]*f2_S[Kcidx])
    push!(c1vec, maximum(r1_S[Kcidx] ./ w1vec)+tol)
    push!(c2vec, maximum(r2_S[Kcidx] ./ w2vec)+tol)

  end

  c1vec = Array{Float64}(c1vec)
  c2vec = Array{Float64}(c2vec)

  if flag == false
    throw(ArgumentError("tmin = tmax"))
  end
  mb = get_momentbounds(P, r1_S, f1_S, c1vec, trange)
  lo = LyapunovMultiKOut(minKidx, minKcidx, f1_S, f2_S, r1_S, r2_S, c1vec, c2vec, tmin, tminidx, trange, mb)
  return lo
end




function get_all_K(mc::MarkovChain,
                   r1::Function, f1::Function,
                   r2::Function, f2::Function,
                   trange::AbstractArray, w::Function;
                   tol=1e-12)

  P = mc.P
  n = size(P,1)
  f1_S = [f1(ind_to_vec(mc.vss,x)) for x in 1:n]
  f2_S = [f2(ind_to_vec(mc.vss,x)) for x in 1:n]
  r1_S = [r1(ind_to_vec(mc.vss,x)) for x in 1:n]
  r2_S = [r2(ind_to_vec(mc.vss,x)) for x in 1:n]

  return get_all_K(mc, r1_S, f1_S, r2_S, f2_S, trange, tol=tol)
end





function get_all_K(mc::MarkovChain,
                   r1_S::AbstractArray, f1_S::AbstractArray,
                   trange::AbstractArray, w::Function; 
                   tol=1e-12)

  P = mc.P
  n = size(P,1)

  boolvec = (P*f1_S.<f1_S) 
  if all(boolvec.==0)
    throw(ArgumentError("No K available for this Lyapunov function."))
  end

  c1vec = []
  c2vec = []
  tmin = 0
  flag = false
  minKidx = []
  minKcidx = []
  tminidx = 0

  Aidx = collect(1:n)

  for (i, ti) in enumerate(trange)
    Kidx = get_sublevelset(mc.vss, w, ti)
    if !(issubset(Kidx, Aidx))
      break
    end
    Kcidx = setdiff(Aidx, Kidx)
    if any((P[Kcidx,Kcidx]*f1_S[Kcidx]).>=f1_S[Kcidx])
        continue
    elseif flag==false
      tmin = ti
      tminidx = i
      minKidx = Kidx
      minKcidx = Kcidx
      flag = true
    end

    w1vec = f1_S[Kcidx] - P[Kcidx,Kcidx]*f1_S[Kcidx]
    push!(c1vec, maximum(r1_S[Kcidx] ./ w1vec)+tol)
  end

  c1vec = Array{Float64}(c1vec)

  if flag == false
    throw(ArgumentError("tmin = tmax"))
  end
  mb = get_momentbounds(P, r1_S, f1_S, c1vec, trange)
  lo = (minKidx=minKidx, 
        minKcidx=minKcidx,
        f1_S=f1_S, 
        r1_S=r1_S,
        c1vec=c1vec, 
        tmin=tmin, 
        tminidx=tminidx, 
        trange=trange, 
        mb=mb)
  return lo
end


function get_sublevel_set_K(
  mc::MarkovChain,
  r1_S::Vector{Float64},
  f1_S::Vector{Float64},
  r2_S::Vector{Float64},
  f2_S::Vector{Float64},
  w::Function,
  trange::Vector{Int64}
)
  P = mc.P
  n = size(P,1)

  boolvec = (P*f1_S.<f1_S) .& (P*f2_S.<f2_S)
  if all(boolvec.==0)
    throw(ArgumentError("No K available for this Lyapunov function."))
  end
  Aidx = collect(1:n)
  for (i, ti) in enumerate(trange)
    Kidx = get_sublevelset(mc.vss, w, ti)
    if !(issubset(Kidx, Aidx))
      break
    end
    Kcidx = setdiff(Aidx, Kidx)
    if (any((P[Kcidx,Kcidx]*f1_S[Kcidx])>=f1_S[Kcidx]) ||
        any((P[Kcidx,Kcidx]*f2_S[Kcidx])>=f2_S[Kcidx]))
        continue
    else
      w1vec = (f1_S[Kcidx] - P[Kcidx, Kcidx]*f1_S[Kcidx])
      w2vec = (f2_S[Kcidx] - P[Kcidx, Kcidx]*f2_S[Kcidx])
      return maximum(r1_S[Kcidx] ./ w1vec), maximum(r2_S[Kcidx] ./ w2vec), Kidx
    end
  end
end



function find_sublevel_tmin(mc::MarkovChain, g::Function, r::Function, w::Function; tmax=100)
  g_S = vss_func_to_vec(mc.vss, g)
  r_S = vss_func_to_vec(mc.vss, r)
  Sidx = collect(1:mc.vss.size)
  
  for t=0:tmax
    Kidx = get_sublevelset(mc.vss, w, t)
    Kcidx = setdiff(Sidx, Kidx)
    if all((mc.P-I)[Kcidx,Kcidx]*g_S[Kcidx].<= -r_S[Kcidx])
      return t
    end
  end
  throw(ArgumentError("tmax not high enough or function not Lyapunov function."))
end


# This code can be optimized.
function find_smallest_K(
  mc::MarkovChain,
  g_S::AbstractArray,
  r_S::AbstractArray,
  w::Function,
  tmin::Int64,
  tmax::Int64,
  )

  Sidx = collect(1:mc.vss.size)
  Kmin =  get_midlevel_set(mc.vss, w, tmin, tmin+1)
  if is_Lyapunov_function(mc.P, setdiff(Sidx, Kmin), r_S, g_S)
    throw(ArgumentError("tmin may be too small."))
  end

  flag = false;

  for t1 in tmin:tmax
    for t2 in tmin:tmax
      if t2>t1
        continue
      end
      
      Kidx = get_midlevel_set(mc.vss, w, t1, t2)
      Kcidx = setdiff(Sidx, Kidx)

      if is_Lyapunov_function(mc.P, Kcidx, r_S, g_S)
        temp = size(Kidx,1) 
        if flag
          if temp < nk
            nk = temp
            Kmin = Kidx
            tminstar = t2
            tmaxstar = t1
          end
        else
          nk = size(Kidx,1)
          Kmin = Kidx
          tminstar = t2
          tmaxstar = t1
          flag = true
        end
      end
    end
  end

  if flag == false
    throw(ArgumentError("Not large enough t-window"))
  end

  return Kmin, tminstar, tmaxstar
  
end

# Views
function is_Lyapunov_function_rate_view(
  Q::AbstractArray,
  Kc::Vector{Int64},
  r::AbstractArray,
  g::AbstractArray
  )
  if all(@views (Q[Kc,Kc]*g[Kc].<=-r[Kc]))
    return true
  else 
    return false
  end
end





function get_K_rate_matrix_heuristic(
  Q::AbstractArray,
  r_S::AbstractArray,
  f_S::AbstractArray
)
  n = size(Q,1)
  Sidx = collect(1:n)
  tup = get_K_rate_matrix_simple(Q, r_S, f_S)
  idx = sortperm(tup.v)
  KC = tup.KC
  i=1
  while true
	  if tup.v[idx[i]]<= 0
		  i+=1
	  else
      if is_Lyapunov_function_rate(Q, union(KC, idx[i]), r_S, f_S)
        KC = union(KC, idx[i])
        i+=1
      else
			  break
		  end
	  end
  end
  return (K=setdiff(Sidx, KC), KC=KC)
end

# Using views didn't seem to add improvement.
function get_K_rate_matrix_heuristic_view(
  Q::AbstractArray,
  r_S::AbstractArray,
  f_S::AbstractArray
)
  n = size(Q,1)
  Sidx = collect(1:n)
  tup = get_K_rate_matrix_simple(Q, r_S, f_S)
  idx = sortperm(tup.v)
  KC = tup.KC
  i=1
  while true
	  if tup.v[idx[i]]<= 0
		  i+=1
	  else
      if is_Lyapunov_function_rate_view(Q, union(KC, idx[i]), r_S, f_S)
        KC = union(KC, idx[i])
        i+=1
      else
			  break
		  end
	  end
  end
  return (K=setdiff(Sidx, KC), KC=KC)
end




function subset_loop(
  Q::AbstractArray,
  r_S::AbstractArray,
  g_S::AbstractArray,
  Kstart::AbstractArray,
  ordering::Function
  )
  Sidx = collect(1:size(Q,1))
  Kc = setdiff(Sidx, Kstart)

  for (i,idx) in enumerate(Kstart)
    temp = union(Kc, idx)
    if is_Lyapunov_function_rate(Q, union(Kc, idx), r_S, g_S)
      Kc = temp
    end
  end
  return setdiff(Sidx, Kc)
end

function subset_loop(
  Q::AbstractArray,
  r_S::AbstractArray,
  g_S::AbstractArray,
  Kstart::AbstractArray
  )
  Sidx = collect(1:size(Q,1))
  Kc = setdiff(Sidx, Kstart)

  for (i,idx) in enumerate(Kstart)
    temp = union(Kc, idx)
    if is_Lyapunov_function_rate(Q, union(Kc, idx), r_S, g_S)
      Kc = temp
    end
  end
  return setdiff(Sidx, Kc)
end

function subset_loop(
  Q::AbstractArray, 
  r1_S::AbstractArray,
  g1_S::AbstractArray,
  r2_S::AbstractArray,
  g2_S::AbstractArray,
  Kstart::AbstractArray)

  Sidx = collect(1:size(Q,1))
  Kc = setdiff(Sidx, Kstart)

  for (i,idx) in enumerate(Kstart)
    temp = union(Kc, idx)
    if is_Lyapunov_function_rate(Q, temp, r1_S, g1_S) && is_Lyapunov_function_rate(Q, temp, r2_S, g2_S)
      Kc = temp
    end
  end
  return setdiff(Sidx, Kc)
end

function get_K_quickstop(
  Q::AbstractArray,
  r1_S::AbstractArray,
  g1_S::AbstractArray,
  r2_S::AbstractArray,
  g2_S::AbstractArray;
  count_max::Int64=1,
)
  n = size(Q,1)
  Sidx = collect(1:n)
  tup_1 = get_K_rate_matrix_simple(Q, r1_S, g1_S)
  tup_2 = get_K_rate_matrix_simple(Q, r2_S, g2_S)
  idx = sortperm(min.(tup_1.v,tup_2.v))
  KC = intersect(tup_1.KC, tup_2.KC)
  i=1
  count = 0
  while count<count_max
	  if (tup_1.v[idx[i]] <= 0) && (tup_2.v[idx[i]] <= 0)
		  i+=1
      # println("here")
	  else
      # println("here")
      try_KC = union(KC, idx[i])
      if is_Lyapunov_function_rate(Q, try_KC, r1_S, g1_S) && 
         is_Lyapunov_function_rate(Q, try_KC, r2_S, g2_S)
        KC = try_KC
        i+=1
      else
			  count = count +1
		  end
	  end
  end
  return (K=setdiff(Sidx, KC), KC=KC)
end

function get_K_tandem(
  Q::AbstractArray,
  r1_S::Vector{Float64},
  g1_S::Vector{Float64},
  r2_S::Vector{Float64},
  g2_S::Vector{Float64}
  )

  lo_1 = get_K_rate_matrix_simple(Q, r1_S, g1_S)
  lo_2 = get_K_rate_matrix_simple(Q, r2_S, g2_S)
  return subset_loop(Q, r1_S, g1_S, r2_S, g2_S, union(lo_1.K, lo_2.K))
end

function get_K_standard_and_subset(
  Q::AbstractArray,
  r1_S::Vector{Float64},
  g1_S::Vector{Float64})

  lo_1 = get_K_rate_matrix_simple(Q, r1_S, g1_S)
  K_1 = subset_loop(Q, r1_S, g1_S, lo_1.K)
  return K_1
end


# Thought this would be faster but looks like no.
function ordered_subset_loop(
  Q::AbstractArray,
  r_S::AbstractArray,
  g_S::AbstractArray,
  Kstart::AbstractArray,
  mincard::Int64,
  ordering::Function;
  )

  Sidx = collect(1:size(Q,1))
  n = size(Kstart,1)
  K_curr = sort(Kstart,by=ordering)
  Kc_curr = setdiff(Sidx, K_curr)
  
  while n > mincard
    for i=1:n
      Kc_temp = union(Kc_curr, K_curr[i])
      if is_Lyapunov_function_rate(Q, Kc_temp, r_S, g_S)
        K_curr = setdiff(K_curr, K_curr[i])
        Kc_curr = Kc_temp
        n=n-1
        break
      end
    end
  end
  return K_curr
end


function get_K_rate_matrix_heuristic2(
  Q::AbstractArray,
  r_S::AbstractArray,
  f_S::AbstractArray,
  mincard::Int64  
  )
  Sidx = collect(1:size(Q,1))
  lo = get_K_rate_matrix_heuristic(Q, r_S, f_S)
  K_h2 = subset_loop(Q, r_S, f_S, lo.K, mincard)
  if !is_Lyapunov_function_rate(Q, setdiff(Sidx, K_h2), r_S, f_S)
    error("Not a Lyapunov function.")
  end
  return K_h2
end

# Deprecated: Simpler method seems to perform better,
# function subset_loop(
  #   Q::AbstractArray,
  #   r_S::AbstractArray,
  #   g_S::AbstractArray,
  #   Kstart::AbstractArray,
  #   mincard::Int64
  #   )
  
  #   Sidx = collect(1:size(Q,1))
  #   n = size(Kstart,1)-1
  #   Kcurr = Kstart
    
  #   while n>= mincard
  #     iter = powerset(Kcurr, n)
  #     t = 0
  #     for temp in iter
  #       Kc = setdiff(Sidx, temp)
  #       if is_Lyapunov_function_rate(Q, Kc, r_S, g_S)
  #         Kcurr = temp
  #         n=n-1
  #         break
  #       end
  #       t = t+1
  #     end
  #     if t == n+1
  #       return Kcurr
  #     end
  #   end
  
  #   return Kcurr
  # end
  