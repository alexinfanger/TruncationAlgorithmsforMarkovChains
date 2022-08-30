"""
		get_strict_interior(mc::GM1K, r::Number)

Return ``\\mathcal{N}_r`` (see [1]), the set of states that are not accesible 
in one jump from outside a truncation set ``A,`` (here assumed to be an r-sublevel
set). In other words, ``\\mathcal{N}_r`` is the complement of the in-boundary 
of ``A`` restricted to ``A''. One might call this the "strict interior" of A with
respect to the geometry induced by considering single jumps via P.

The current implementation assumes ``A`` is an r-sublevel set of the function
``f(x)=x_1+x_2+..x_d`` in dimension ``d``.
"""
function get_strict_interior(mc::GM1K, r::Number)
	return []
end

function get_strict_interior(mc::ReflectedToggleSwitch, r::Number)
	if r==0 
		return []
	else
		return get_linear_sublevel_set(mc.vss, r-1)
	end
end


# function get_strict_interior(mc::MM1K, r::Number)
# 	return get_linear_sublevel_set(mc.vss, r-1)
# end

# function get_strict_interior(mc::OpenJacksonNetwork, r::Number)
# 	return get_linear_sublevel_set(mc.vss, r-1)
# end
