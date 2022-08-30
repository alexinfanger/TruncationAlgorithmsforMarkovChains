using LinearAlgebra

struct VectorStateSpace
    ndims::Int64
    array_of_sizes::Vector{Int64}
    expansion_vals::Vector{Int64}
    size::Int64
end

function VectorStateSpace(array_of_sizes::AbstractArray)
    k = size(array_of_sizes,1)
    expansion_vals = zeros(Int64, k)
    for i=1:k
        expansion_vals[i] = reduce(*,array_of_sizes[2+(i-1):end])
    end
    return VectorStateSpace(k, array_of_sizes,
                expansion_vals, expansion_vals[1]*array_of_sizes[1])
end

function vec_to_ind(vss::VectorStateSpace, vec::AbstractArray)
    ind = 1
    for i=1:vss.ndims
        ind += (vec[i]-1)*vss.expansion_vals[i]
    end
    return ind
end

function ind_to_vec(vss::VectorStateSpace, ind::Integer)
    ind = ind-1
    vec = zeros(Integer, vss.ndims)
    curr = ind
    for i=1:vss.ndims
        vec[i] = floor(curr/vss.expansion_vals[i])
        curr = rem(curr,vss.expansion_vals[i])
    end
    return vec.+1
end

function ind_to_vec(vss::VectorStateSpace, indarr::AbstractArray)
    out = Array{AbstractArray}(undef, size(indarr))
    for (i,t) in enumerate(indarr)
        ind_to_vec(vss, t)
        out[i] = ind_to_vec(vss, t)
    end
    return out
end

function func_to_array(vss::VectorStateSpace, vec::AbstractArray, vecidx::AbstractArray)
    n = size(vec,1)
    arr = zeros(n,vss.ndims+1)
    for (i,t) in enumerate(vecidx)
        arr[i,1:end-1] = ind_to_vec(vss,t)
        arr[i,end] = vec[i]
    end
    return arr
end

function TwoDMatrix(vss::VectorStateSpace, vec::AbstractArray, vecidx::AbstractArray)
    zmat = zeros(vss.array_of_sizes[1],vss.array_of_sizes[2])
    for (i,t) in enumerate(vecidx)
        zmat[Integer(ind_to_vec(vss,t)[1]),Integer(ind_to_vec(vss,t)[2])]=vec[i]
    end
    return zmat
end

function vec_to_twodmatrix(vss::VectorStateSpace, vec::AbstractArray)
    zmat = zeros(vss.array_of_sizes[1],vss.array_of_sizes[2])
    for i=1:size(vec,1)
        vecind = ind_to_vec(vss,i)
        zmat[Integer(vecind[1]),Integer(vecind[2])]=vec[i]
    end
    return zmat
end

function get_sublevelset(n::Number, f::Function, t::Number)
    sublevelset = Array{Int64, 1}(UndefInitializer(), 0)
    for i=1:n
        if f(i.-1)<=t
            append!(sublevelset,i)
        end
    end
    return sublevelset
end

function get_sublevelset(vss::VectorStateSpace, p::Number, r::Number)
    sublevelset = Array{Int64, 1}(UndefInitializer(), 0)
    for i=1:vss.size
        if norm(ind_to_vec(vss, i).-1,p)<=r
            append!(sublevelset,i)
        end
    end
    return sublevelset
end

function get_sublevelset(vss::VectorStateSpace, f::Function, t::Number)
    sublevelset = Array{Int64, 1}(UndefInitializer(), 0)
    for i=1:vss.size
        if f(ind_to_vec(vss,i))<=t
            append!(sublevelset,i)
        end
    end
    return sublevelset
end

function get_levelset(vss::VectorStateSpace, f::Function, t::Number)
    levelset = Array{Int64, 1}(UndefInitializer(), 0)
    for i=1:vss.size
        if f(ind_to_vec(vss,i))==t
            append!(levelset,i)
        end
    end
    return levelset
end

function get_sublevelset_zero(vss::VectorStateSpace, p::Number, r::Number)
    sublevelset = Array{Int64, 1}(UndefInitializer(), 0)
    for i=1:vss.size
        if norm(ind_to_vec(vss, i).-1,p)<=r
            append!(sublevelset,i)
        end
    end
    return sublevelset
end


function get_linear_sublevel_set(vss::VectorStateSpace, t::Number)
    e  = ones(vss.ndims)
    return get_sublevelset(vss, x->e'*x.-vss.ndims, t)
end

function get_linear_level_set(vss::VectorStateSpace, t::Number)
    e  = ones(vss.ndims)
    return get_levelset(vss, x->e'*x.-vss.ndims, t)
end

function get_midlevel_set(
    vss::VectorStateSpace, 
    w::Function,
    t1::Number,
    t2::Number)
    tmax = max(t1,t2)
    tmin = min(t1,t2)
    sublevel_big = get_sublevelset(vss, w,tmax)
    sublevel_small = get_sublevelset(vss, w, tmin)
    return setdiff(sublevel_big, sublevel_small)
end

function zeropad(pi_approx::AbstractArray, idx::AbstractArray, vss::VectorStateSpace)
    zp = zeros(1,vss.size)
    zp[idx] = pi_approx
    return zp
end

function vss_abspow_function(vss::VectorStateSpace, p::Number)
    d = vss.ndims
    return y->sum([(abs(y[i]-1))^p for i=1:d])
end

function vss_centered_abspow_function(vss::VectorStateSpace, p::Number, c::AbstractArray)
    if size(c,1) != vss.ndims
        throw(ArgumentError("Centering vector has wrong dimensions."))
    end
    d = vss.ndims
    return y->sum([(abs(y[i]-c[i]))^p for i=1:d])
end

function vss_sumpow_function(vss::VectorStateSpace, p::Number)
    d = vss.ndims
    return y->(sum([(y[i]-1) for i=1:d]))^p
end

function vss_func_to_vec(vss::VectorStateSpace, f::Function)
    fvec = zeros(vss.size)
    for i = 1:vss.size
        fvec[i]= f(ind_to_vec(vss,i))
    end
    return fvec
end

function vss_abspow_vec(vss::VectorStateSpace, p::Int64)
    return vss_func_to_vec(vss, vss_abspow_function(vss,p))
end

function vss_centered_abspow_vec(vss::VectorStateSpace, p::Int64, c::AbstractArray)
    return vss_func_to_vec(vss, vss_centered_abspow_function(vss, p, c))
end

function vss_sumpow_vec(vss::VectorStateSpace, p::Int64)
    return vss_func_to_vec(vss, vss_sumpow_function(vss, p))
end

function Aidx_to_Sidx(
    vss::VectorStateSpace, 
    vec::AbstractArray,
    Aidx::AbstractArray)
    full_vec = zeros(vss.size)
    full_vec[Aidx] = vec
    return full_vec
  end

 function vss_heatmap(vss::VectorStateSpace, vec_S::Vector; maxsize::Int64 =100)
    if vss.ndims !=2
        throw(ArgumentError("Heatmap only for 2d vector state space."))
    end
    p = heatmap(vec_to_twodmatrix(vss,vec_S)[1:maxsize,1:maxsize])
    return p
 end

function vss_idx_to_boolvec(vss::VectorStateSpace, idx::Vector{Int64})
    boolvec = zeros(vss.size)
    boolvec[idx] .= 1.0
    return boolvec
end

function vss_idx_to_heatmap(vss::VectorStateSpace, idx::Vector{Int64}; maxsize::Int64 =100)
    boolvec = vss_idx_to_boolvec(vss, idx)
    return vss_heatmap(vss, boolvec, maxsize=maxsize)
end

function vss_2d_symmetric_coordinate(vss::VectorStateSpace, idx::Int64)
    x,y = ind_to_vec(vss, idx)
    return vec_to_ind(vss, [y,x])
end

function vss_2d_symmetrize_to_zero(
    vss::VectorStateSpace,
    idx::AbstractArray)
    newidx = copy(idx)
    for i in idx
        if !(vss_2d_symmetric_coordinate(vss, i) in idx)
            setdiff!(newidx, i)
        end
    end
    return newidx
end

function vss_2d_is_symmetric(
  vss::VectorStateSpace,
  idx::Vector{Int64})
  for i in idx
    j = vss_2d_symmetric_coordinate(vss, i)
    if !(j in idx)
      return false
    end
  end
  return true
end