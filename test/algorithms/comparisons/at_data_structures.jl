#= Data structures for at ("all tests")
   @pre Assumes local operator P.
   @pre Assumes sublevel and level sets used as truncation sets.
   @pre Reward function assumed to be non-negative.
=#

FixedKRunInput = @NamedTuple begin
    tvals::Vector{Int64}
    Kidx::Vector{Int64}
    r_S::Vector{Float64}
    f1_S::Vector{Float64}
    f2_S::Vector{Float64}
    name::String
end

AlgFlags = @NamedTuple begin
  lpoa::Bool
  rta::Bool
  lp::Bool
  ptb::Bool
  CS::Bool
end

MomentBound = @NamedTuple{
  active::Bool,
  bound::Float64,
  tvals::Vector{Int64},
  w_S::Vector{Float64},
  # r_over_w_vec_T::Vector{Float64},
  name::String
}

MomentBounds = @NamedTuple{
  SDP::MomentBound, 
  Lyapunov::MomentBound, 
  # Lyapunov2::MomentBound,
  exact::MomentBound
}

ATRewardFunction = @NamedTuple{
    r_S::Vector{Float64},
    pir_trueval::Float64,
    f1_S::Vector{Float64},
    name::String,
    fk_ins::Dict{String,FixedKRunInput},
    tvals::Vector{Int64},
    r_over_w::Vector{Float64},
    w_tvals::Vector{Int64},
    algflags::AlgFlags,
    momentbounds::MomentBounds,
    Lyapunov_out::LyapunovMultiKOut,
    Lyapunov_time::Float64,
    mb_time::Float64
}

# function init_ATRewardFunction(
#   r_S::Vector{Float64}, 
#   f1_S::Vector{Float64},
#   pi_S::Vector{Float64},
#   name::String,
#   FixedKs::Vector{FixedKRunInput},
#   tvals::Vector{Int64},
#   r_over_w::Vector{Float64},
#   w_tvals::Vector{Int64},
#   lpoa::Bool,
#   rta::Bool,
#   lp::Bool,
#   pert::Bool
#   )
#   return At(r_S=r_S, 
#           pir_trueval=pi_S'*r_S,
#           f1_S=f1_S,
#           name=name,
#           FixedKs=FixedKs,
#           tvals=tvals,
#           r_over_w=r_over_w,
#           lpoa=lpoa,
#           rta=rta,
#           lp=lp,
#           pert=pert
#           )
# end


MomentBoundBased = @NamedTuple{
  SDP::Vector{Float64},
  Lyapunov::Vector{Float64},
  # Lyapunov2::Vector{Float64},
  exact::Vector{Float64}
}

KuntzBoundBased = @NamedTuple{
  mb::MomentBoundBased,
  tb::Vector{Float64}
}

function init_KuntzBoundBased(nt::Int64)
  return (mb=(SDP = zeros(nt)*NaN,
              Lyapunov = zeros(nt)*NaN,
              exact = zeros(nt)*NaN),
          tb=zeros(nt)*NaN)
end

# function MomentBoundBased(nt::Number)
    # return (SDP= zeros(nt),
    #         Lyapunov = zeros(nt),
    #         exact = zeros(nt),
    #         lowerbound = zeros(nt))
# end

ATIn = @NamedTuple {
  pcmc::PrecomputedMarkovProcess,
  rs::Dict{String, ATRewardFunction},
  fixedKnames::Vector{String},
  tvals::Vector{Int64},
  # w_S::Vector{Float64},
  # w_tvals::Vector{Int64},
  momentbounds::MomentBounds,
  tailmass_bound::Bool,
  tailmass_vec::Vector{Float64},
  run_levelsets::Bool,
  benchmark::Bool
}

ATApxOut_i = @NamedTuple begin
  apx_errs::Matrix{Float64}
  apx_times::Matrix{Float64}
  apx_memory::Matrix{Int64}
end

function init_ATApxOut_i(nt::Integer)
  return (apx_errs = zeros(nt,nt)*NaN,
          apx_times = zeros(nt,nt)*NaN,
          apx_memory = zeros(nt,nt))
end

ATApxOut = @NamedTuple begin
  rn::ATApxOut_i
  pf::ATApxOut_i
  cond::ATApxOut_i
  pi3::ATApxOut_i
end


function init_ATApxOut(nt::Integer)
  return (rn=init_ATApxOut_i(nt), 
          pf=init_ATApxOut_i(nt),
          cond=init_ATApxOut_i(nt), 
          pi3=init_ATApxOut_i(nt))
end

ATrApxOut_i = @NamedTuple begin
    r_apx_values::Matrix{Float64}
    r_apx_times::Matrix{Float64}
    r_apx_memory::Matrix{Int64}
end

function init_ATrApxOut_i(nt::Integer)
    return (r_apx_values=zeros(nt,nt)*NaN, 
            r_apx_times=zeros(nt,nt)*NaN, 
            r_apx_memory=zeros(nt,nt))
end

ATrApxOut = @NamedTuple begin
    rn::ATrApxOut_i
    pf::ATrApxOut_i
    cond::ATrApxOut_i
    pi3::ATrApxOut_i
end

function init_ATrApxOut(nt::Integer)
    return (rn=init_ATrApxOut_i(nt), 
            pf=init_ATrApxOut_i(nt), 
            cond=init_ATrApxOut_i(nt), 
            pi3=init_ATrApxOut_i(nt))
end

ATrBdsOut_i = @NamedTuple begin
    rweightedTVs::Matrix{Float64}
    rweightedTVstimes::Matrix{Float64}
    ptb_lbs::Matrix{Float64}
    ptb_ubs::Matrix{Float64}
    ptb_bds_times::Matrix{Float64}
    ptb_bds_memory::Matrix{Int64}

    cond_Finvs::Matrix{Float64}
    Deltas::Matrix{Float64}
    minrowsum_Finvs::Matrix{Float64}
    maxrowsum_Finvs::Matrix{Float64}
end

function init_ATrBdsOut_i(nt::Integer)
    return (rweightedTVs=zeros(nt,nt)*NaN, 
            rweightedTVstimes=zeros(nt,nt)*NaN, 
            ptb_lbs=zeros(nt,nt)*NaN, 
            ptb_ubs=zeros(nt,nt)*NaN,
            ptb_bds_times=zeros(nt,nt)*NaN,
            ptb_bds_memory=zeros(nt,nt),
            cond_Finvs=zeros(nt, nt)*NaN, 
            Deltas=zeros(nt,nt)*NaN, 
            minrowsum_Finvs=zeros(nt, nt),
            maxrowsum_Finvs=zeros(nt, nt))
end

ATIOrOut = @NamedTuple begin
    r_apx_out::ATrApxOut
    rn::ATrBdsOut_i
    pf::ATrBdsOut_i
    lp_lb::Matrix{Float64}
    lp_ub::Matrix{Float64}
    lp_times::Matrix{Float64}
    lp_memory::Matrix{Int64}
end

function init_ATIOrOut(nt::Integer)
    return (r_apx_out=init_ATrApxOut(nt), 
            rn=init_ATrBdsOut_i(nt),
            pf=init_ATrBdsOut_i(nt), 
            lp_lb=zeros(nt,nt)*NaN,
            lp_ub=zeros(nt,nt)*NaN, 
            lp_times=zeros(nt,nt)*NaN,
            lp_memory=zeros(nt,nt))
end

ATLPOAOut = @NamedTuple begin
    approx_tvs::KuntzBoundBased
    times::Vector{Float64}
    memory::Vector{Int64}
    tvs_with_conditional::Vector{Float64}
end

function init_ATLPOAOut(nt::Integer)
    return (        
      approx_tvs=init_KuntzBoundBased(nt), 
      times=zeros(nt)*NaN,
      memory=zeros(nt),
      tvs_with_conditional=zeros(nt)*NaN
      )
end

ATLPOArOut = @NamedTuple begin
    lbs::KuntzBoundBased
    ubs::KuntzBoundBased
    lbs_r::Vector{Float64}
    ubs_r::Vector{Float64}
    approx_errs::Vector{Float64}
    times::Vector{Float64}
    memory::Vector{Int64}
end

function init_ATLPOArOut(nt::Integer)
    return (
              lbs = init_KuntzBoundBased(nt),
              ubs = init_KuntzBoundBased(nt), 
              lbs_r = zeros(nt)*NaN,
              ubs_r = zeros(nt)*NaN,
              approx_errs=zeros(nt)*NaN, 
              times=zeros(nt)*NaN,
              memory=zeros(nt)
            )
end

ATLPOA = @NamedTuple begin
    dist::ATLPOAOut
    rs::Dict{String,ATLPOArOut}
end

function init_ATLPOA(
  nt::Integer, 
  rnames::Vector{String},
  rnts::Vector{Int64})
    rd = Dict{String,ATLPOArOut}()
    for (rn, rnt) in zip(rnames, rnts)
        rd[rn] = init_ATLPOArOut(rnt)
    end
    return ATLPOA((init_ATLPOAOut(nt), rd))
end

ATRTAOut = @NamedTuple begin
    approx_tvs::Vector{Float64}
    tv_bound_lowers::KuntzBoundBased
    tv_bound_uppers::KuntzBoundBased
    approx_times::Vector{Float64}
    approx_memory::Vector{Int64}
    tvs_with_conditional::Vector{Float64}
end

function init_ATRTAOut(nt::Integer)
    return (approx_tvs=zeros(nt)*NaN, 
            tv_bound_lowers=init_KuntzBoundBased(nt), 
            tv_bound_uppers=init_KuntzBoundBased(nt), 
            approx_times=zeros(nt)*NaN, 
            approx_memory=zeros(nt),
            tvs_with_conditional=zeros(nt)*NaN)
end


# For simplicity, we assume that the functions we 
# are interested in are non-negative, so only the upper-bounds
# will have dependence on the moment bound
ATRTArOut = @NamedTuple begin
    approx_errs::Vector{Float64}
    lbs::KuntzBoundBased
    ubs::KuntzBoundBased
    conditional_lbs::Vector{Float64}
    conditional_ubs::Vector{Float64}
end

function init_ATRTArOut(nt::Integer)
    return (approx_errs=zeros(nt)*NaN,
            lbs=init_KuntzBoundBased(nt), 
            ubs=init_KuntzBoundBased(nt),
            conditional_lbs=zeros(nt)*NaN,
            conditional_ubs=zeros(nt)*NaN
            )
end

ATRTA = @NamedTuple begin
    dist::ATRTAOut
    rs::Dict{String,ATRTArOut}
end

function init_ATRTA(
  nt::Integer,
  rnames::Vector{String},
  rnts::Vector{Int64})
    rd = Dict{String,ATRTArOut}()
    for (rnm, rnt) in zip(rnames,rnts)
        rd[rnm] = init_ATRTArOut(rnt)
    end
    return ATRTA((init_ATRTAOut(nt),rd))
end


ATIOOut = @NamedTuple begin
    routs::Dict{String,ATIOrOut}
    apx_out::ATApxOut
    rho_Gs::Matrix{Float64}
    minrowsum_Gs::Matrix{Float64}
    condImP22s::Matrix{Float64}
    level_name::String
end

function init_ATIOOut(
  routs::Dict{String,ATIOrOut},
  nt::Integer,
  levelname::String)
    return ATIOOut((routs, 
     init_ATApxOut(nt), 
     zeros(nt,nt)*NaN, 
     zeros(nt,nt)*NaN, 
     zeros(nt,nt)*NaN,
     levelname))
end


ATExit = @NamedTuple begin
    dist::ATApxOut_i
    rs::Dict{String,ATrApxOut_i}
end

function init_ATExit(
  nt::Int64,
  rnames::Vector{String},
  rnts::Vector{Int64})
    rd = Dict{String,ATrApxOut_i}()
    for (rnm,rnt) in zip(rnames,rnts)
        rd[rnm] = init_ATrApxOut_i(rnt)
    end
    return ATExit((init_ATApxOut_i(nt), rd))
end

FixedKApxOut_i = @NamedTuple begin
    apx_errs::Vector{Float64}
    apx_times::Vector{Float64}
    apx_memory::Vector{Int64}
  end  

  function init_FixedKApxOut_i(nt::Integer)
    return (apx_errs = zeros(nt)*NaN,
            apx_times = zeros(nt)*NaN,
            apx_memory = Vector{Int64}(zeros(nt)))
  end
  
  FixedKApxOut = @NamedTuple begin
    rn::FixedKApxOut_i
    pf::FixedKApxOut_i
    cond::FixedKApxOut_i
    pi3::FixedKApxOut_i
    CS_weighted::FixedKApxOut_i
    CS_min_tv::FixedKApxOut_i
  end
  
  
  function init_FixedKApxOut(nt::Integer)
    return (rn=init_FixedKApxOut_i(nt), 
            pf=init_FixedKApxOut_i(nt),
            cond=init_FixedKApxOut_i(nt), 
            pi3=init_FixedKApxOut_i(nt),
            CS_weighted=init_FixedKApxOut_i(nt),
            CS_min_tv=init_FixedKApxOut_i(nt))
  end
  
  FixedKrApxOut_i = @NamedTuple begin
      r_apx_values::Vector{Float64}
      r_apx_times::Vector{Float64}
      r_apx_memory::Vector{Int64}
  end
  
  function init_FixedKrApxOut_i(nt::Integer)
      return (r_apx_values=zeros(nt)*NaN, 
              r_apx_times=zeros(nt)*NaN, 
              r_apx_memory=Vector{Int64}(zeros(nt)))
  end
  
  FixedKrApxOut = @NamedTuple begin
      rn::FixedKrApxOut_i
      pf::FixedKrApxOut_i
      cond::FixedKrApxOut_i
      pi3::FixedKrApxOut_i
  end
  
  function init_FixedKrApxOut(nt::Integer)
      return (rn=init_FixedKrApxOut_i(nt), 
              pf=init_FixedKrApxOut_i(nt), 
              cond=init_FixedKrApxOut_i(nt), 
              pi3=init_FixedKrApxOut_i(nt))
  end
  
  FixedKrBdsOut_i = @NamedTuple begin
      rweightedTVs::Vector{Float64}
      rweightedTVstimes::Vector{Float64}
      ptb_lbs::Vector{Float64}
      ptb_ubs::Vector{Float64}
      ptb_bds_times::Vector{Float64}
      ptb_bds_memory::Vector{Int64}

      CS_ptb_lbs::Vector{Float64}
      CS_ptb_ubs::Vector{Float64}
      CS_rweightedTVs::Vector{Float64}
      CS_rweightedTVstimes::Vector{Float64}
  
      cond_Finvs::Vector{Float64}
      Deltas::Vector{Float64}
      minrowsum_Finvs::Vector{Float64}
      maxrowsum_Finvs::Vector{Float64}
  end
  
  function init_FixedKrBdsOut_i(nt::Integer)
      return (rweightedTVs=zeros(nt)*NaN, 
              rweightedTVstimes=zeros(nt)*NaN, 
              ptb_lbs=zeros(nt)*NaN, 
              ptb_ubs=zeros(nt)*NaN,
              ptb_bds_times=zeros(nt)*NaN,
              ptb_bds_memory=Vector{Int64}(zeros(nt)),
              CS_ptb_lbs=zeros(nt)*NaN,
              CS_ptb_ubs=zeros(nt)*NaN,
              CS_rweightedTVs=zeros(nt)*NaN, 
              CS_rweightedTVstimes=zeros(nt)*NaN, 
              cond_Finvs=zeros(nt)*NaN, 
              Deltas=zeros(nt)*NaN, 
              minrowsum_Finvs=zeros(nt),
              maxrowsum_Finvs=zeros(nt))
  end
  
  FixedKIOrOut = @NamedTuple begin
      r_apx_out::FixedKrApxOut
      rn::FixedKrBdsOut_i
      pf::FixedKrBdsOut_i
      lp_lb::Vector{Float64}
      lp_ub::Vector{Float64}
      lp_times::Vector{Float64}
      lp_memory::Vector{Int64}
      CS_lb::Vector{Float64}
      CS_ub::Vector{Float64}
      CS_times::Vector{Float64}
      CS_heuristic_lb::Vector{Float64}
      CS_heuristic_ub::Vector{Float64}
      CS_heuristic_times::Vector{Float64}
  end
  
  function init_FixedKIOrOut(nt::Integer)
      return (r_apx_out=init_FixedKrApxOut(nt), 
              rn=init_FixedKrBdsOut_i(nt),
              pf=init_FixedKrBdsOut_i(nt), 
              lp_lb=zeros(nt)*NaN,
              lp_ub=zeros(nt)*NaN, 
              lp_times=zeros(nt)*NaN,
              lp_memory=Vector{Int64}(zeros(nt)),
              CS_lb = zeros(nt)*NaN,
              CS_ub = zeros(nt)*NaN,
              CS_times = zeros(nt)*NaN,
              CS_heuristic_lb = zeros(nt)*NaN,
              CS_heuristic_ub = zeros(nt)*NaN,
              CS_heuristic_times = zeros(nt)*NaN                        
              )
  end

FixedKRun = @NamedTuple begin
    # fk_in::FixedKRunInput
    apx_out::FixedKApxOut
    rout::Dict{String,FixedKIOrOut}
end


# # If only doing a single run quickly to test.
# function init_FixedKRun(fk_in::FixedKRunInput)
#     nt = size(fk_in.tvals,1)
#     return (
#             fk_in=fk_in,
#             apx_out = init_FixedKApxOut(nt),
#             rout = init_FixedKIOrOut(nt)
#             )
# end

# For multiple r (and AllTests Data Structure)
function init_FixedKRuns(at_in::ATIn)
  rd = Dict{String,FixedKIOrOut}()  
  for (rname, rval) in at_in.rs
    rnt = size(rval.tvals,1)
    rd[rname] = init_FixedKIOrOut(rnt)
  end
  return (
    apx_out = init_FixedKApxOut(size(at_in.tvals,1)),
    rout = rd
    )
end

ATIO = @NamedTuple begin
    level::ATIOOut
    sublevel::ATIOOut
    fixedK::Dict{String,FixedKRun}
end

function init_ATIO(at_in::ATIn, routs, nt)
  fkd = Dict{String,FixedKRun}()
  for fkname in at_in.fixedKnames
    fkd[fkname]=init_FixedKRuns(at_in)
  end
  return ATIO((init_ATIOOut(routs, nt, "level"), 
              init_ATIOOut(routs, nt, "sublevel"),
              fkd))
end

FixedKRegenerativerOut = @NamedTuple begin
  lbs::Vector{Float64}
  ubs::Vector{Float64}
  eq_exps_bds_times::Vector{Float64}
  rweighted_tvs_bds::Vector{Float64}
  rweighted_tvs_bds_times::Vector{Float64}
end

function init_FixedKRegenerativerOut(rnt)
    return FixedKRegenerativerOut(
      (
        zeros(rnt)*NaN,
        zeros(rnt)*NaN,
        zeros(rnt)*NaN,
        zeros(rnt)*NaN,
        zeros(rnt)*NaN,
       )
    )
end

FixedKRegenerativeRun = @NamedTuple begin
  regenerative_apx::FixedKApxOut_i
  rout::Dict{String,FixedKRegenerativerOut}
end

function init_FixedKRegenerativeRun(at_in::ATIn)
  rd = Dict{String,FixedKRegenerativerOut}()  
  for (rname, rval) in at_in.rs
    rnt = size(rval.tvals,1)
    rd[rname] = init_FixedKRegenerativerOut(rnt)
  end
  return FixedKRegenerativeRun(
    (
    regenerative_apx = init_FixedKApxOut_i(size(at_in.tvals,1)),
    rout = rd
    )
    )
end

ATRegenerative = @NamedTuple begin
  fixedK::Dict{String,FixedKRegenerativeRun}  
end

function init_ATRegenerative(at_in::ATIn)
  fkd = Dict{String,FixedKRegenerativeRun}()
  for fkname in at_in.fixedKnames
    fkd[fkname]=init_FixedKRegenerativeRun(at_in)
  end
  return ATRegenerative((fixedK=fkd,))
end

AT = @NamedTuple begin
    at_in::ATIn
    rinfos::Dict
    IO::ATIO
    rta::ATRTA
    lpoa::ATLPOA
    exit::ATExit
    regenerative::ATRegenerative
end

function init_AT(at_in::ATIn)
    nt = size(at_in.tvals,1)
    routs = Dict{String,ATIOrOut}()
    rinfos = Dict{String, NamedTuple{(:LyapunovOut, :r_S),
                                     Tuple{LyapunovMultiKOut, Vector{Float64}}}}()

    rnames = Vector{String}()
    rnts = Vector{Int64}()

    for (rname, rval) in at_in.rs
      routs[rname] = init_ATIOrOut(size(rval.tvals,1))
      push!(rnames, rname)
      push!(rnts, size(rval.tvals,1))
    end
    return AT((
            at_in, 
            rinfos, 
            init_ATIO(at_in, routs, nt),
            init_ATRTA(nt, rnames, rnts), 
            init_ATLPOA(nt, rnames, rnts),
            init_ATExit(nt, rnames, rnts),
            init_ATRegenerative(at_in)
            ))
end