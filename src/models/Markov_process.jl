abstract type MarkovProcess end

abstract type MarkovChain <: MarkovProcess end
abstract type FiniteMarkovChain <: MarkovChain end
abstract type InfiniteMarkovChain <: MarkovChain end

abstract type MarkovJumpProcess <: MarkovProcess end
abstract type FiniteMarkovJumpProcess <:MarkovJumpProcess end

# Precomputed types just have a .piv and .mc to a Markov chain
abstract type PrecomputedMarkovProcess end
abstract type PrecomputedFiniteMarkovJumpProcess <: PrecomputedMarkovProcess end
abstract type PrecomputedFiniteMarkovChain <: PrecomputedMarkovProcess end


function get_stat_dist(mc::FiniteMarkovChain)
    return get_stat_dist_Q(I-mc.P)
end

function get_stat_dist(mc::PrecomputedFiniteMarkovChain)
    return mc.piv
end


"""
    NamedTuple BoundApproxOut
Data structure for keeping
  - approx: equilibrium expectation approximation
  - lb: lower bound on equilibrium expectation
  - ub: upper bound on equilibrium expectation
  - tv: r-weighted total variation bound
"""
BoundApproxOut = @NamedTuple begin
  approx::Float64
  lb::Float64
  ub::Float64
  tvb::Float64
end

PrecomputedEqExp = @NamedTuple begin
    ptb_out::BoundApproxOut
    name::String
end

