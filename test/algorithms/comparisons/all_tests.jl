# The prefix at stands for "all tests". This is part of a larger code base 
# that not only compares the algorithms, but also computes diagnostic tests.

"""
    at_other_methods_dist!(at) 

This code tests LPOA, RTA, and truncation and augmentation. And in particular,
this tests these methods on computing an entire distribution ``(\\pi(x):x\\in A)``
as opposed to equilibrium expectations.
"""
function at_other_methods_dist!(at) 
  at_in = at.at_in
  piv = at_in.pcmc.piv
  isMJP = at_in.pcmc.mc isa MarkovJumpProcess
  if isMJP
    Q = at_in.pcmc.mc.Q
  elseif at_in.pcmc.mc isa MarkovChain
    P = at_in.pcmc.mc.P
  else 
    throw(ArgumentError("Not MJP or MC."))
  end

  for (i1,t1) in enumerate(at_in.tvals)

    println("Other Methods (dist): t1=", t1)
    Aidx = get_linear_sublevel_set(at_in.pcmc.mc.vss, t1)
    NrIdx = get_strict_interior(at_in.pcmc.mc, t1)
    NrIdx_A = findall(x->x in NrIdx, Aidx)

    cond_dist_S = zeros(at_in.pcmc.mc.vss.size)
    cond_dist_S[Aidx] = piv[Aidx]

    # RTA
    if isMJP
      minvec, maxvec, avgvec = rta_Q(Q, Aidx)
    else
      minvec, maxvec, avgvec = rta(P, Aidx)
    end
    
    minvec_S = Aidx_to_Sidx(at.at_in.pcmc.mc.vss, minvec, Aidx)
    maxvec_S = Aidx_to_Sidx(at.at_in.pcmc.mc.vss, maxvec, Aidx)
    avgvec_S = Aidx_to_Sidx(at.at_in.pcmc.mc.vss, avgvec, Aidx)

    at.rta.dist.tvs_with_conditional[i1] = tv(maxvec_S, cond_dist_S)

    #Lyapunov 
    for (in,l,u) in zip(at.at_in.momentbounds,
                        at.rta.dist.tv_bound_lowers.mb,
                        at.rta.dist.tv_bound_uppers.mb)
      if in.active
        # In [1], the sublevel sets are defined with a strict inequality, 
        # so to use the formulas of [1], we increment our t value defining
        # the sublevel set.
        tk = in.tvals[i1]+1 
        c = in.bound
        l[i1] = 1-sum((1-c/tk)*minvec) # (3.18) in [1]
        u[i1] = max(sum(maxvec)-1+c/tk, c/tk) # (3.20) in [1]
      end
    end

    if at.at_in.tailmass_bound
      tk = at.at_in.tvals[i1]+1
      c = at.at_in.tailmass_vec[i1]*tk
      at.rta.dist.tv_bound_lowers.tb[i1] = 1-sum((1-c/tk)*minvec)
      at.rta.dist.tv_bound_uppers.tb[i1] = min(max(sum(maxvec)-1+c/tk, c/tk), sum(maxvec)-1+2*c/tk)
    end

    at.rta.dist.approx_tvs[i1] = tvzp(maxvec_S, piv)
    if at.at_in.benchmark
      if isMJP
        b = @benchmark rta_Q($Q, $Aidx) 
      else
        b = @benchmark rta($P, $Aidx) 
      end
      at.rta.dist.approx_times[i1] = median(b.times)
      at.rta.dist.approx_memory[i1] = b.memory
    end

    # LPOA
    if !isMJP
      Q = (P-I)
    end
    Q_Aidx = Q[Aidx, Aidx]

    for (in, lpoa) in zip(at.at_in.momentbounds,
                         at.lpoa.dist.approx_tvs.mb)
      if in.active
        println("Warning: only allows for LPOA on a single moment bound for timing.")
        tk = in.tvals[i1]+1
        c = in.bound
        w_Aidx = in.w_S[Aidx]

        piv_LPOA = LPOA(Q_Aidx, NrIdx_A, c, tk, w_Aidx)
        piv_LPOA_S = Aidx_to_Sidx(at.at_in.pcmc.mc.vss, piv_LPOA, Aidx)
        lpoa[i1] = tvzp(piv_LPOA_S, piv)
        at.lpoa.dist.tvs_with_conditional[i1] = tv(piv_LPOA_S, cond_dist_S)
      
        if at.at_in.benchmark
          b = @benchmark LPOA($Q_Aidx, $NrIdx_A, $c, $tk, $w_Aidx) 
          at.lpoa.dist.times[i1] = median(b.times)
          at.lpoa.dist.memory[i1] = b.memory
        end
      end
    end

    if at.at_in.tailmass_bound
      println("Warning: only allows for LPOA on a single moment bound for timing.")
      tk = at.at_in.tvals[i1]+1
      c = at.at_in.tailmass_vec[i1]*tk

      piv_LPOA = LPOA(Q_Aidx, NrIdx_A, c, tk, w_Aidx)
      piv_LPOA_S = Aidx_to_Sidx(at.at_in.pcmc.mc.vss, piv_LPOA, Aidx)
      at.lpoa.dist.approx_tvs.tb[i1] = tvzp(piv_LPOA_S, piv)
    
      if at.at_in.benchmark
        b = @benchmark LPOA($Q_Aidx, $NrIdx_A, $c, $tk, $w_Aidx) 
        at.lpoa.dist.times[i1] = median(b.times)
        at.lpoa.dist.memory[i1] = b.memory
      end
    end


    # Exit approximation (also known as truncation and augmentation)

    if isMJP
      exit_out = exit_approx_Q(Aidx, Q)
    else
      exit_out = exit_approx(Aidx, P)
    end
    at.exit.dist.apx_errs[i1] = tvzp(exit_out.piv_S, piv)
    if at.at_in.benchmark
      if isMJP
        b = @benchmark exit_approx_Q($Aidx, $Q) 
      else
        b = @benchmark exit_approx($Aidx, $P) 
      end
      at.exit.dist.apx_times[i1] = median(b.times)
      at.exit.dist.apx_memory[i1] = b.memory
    end    
  end
end

"""
    at_other_methods_rs!(at) 

This code tests LPOA, RTA, and truncation and augmentation. And in particular,
this is for equilibrium expectations ``\\sum_{x\\in S}\\pi(x)r(x)`` as opposed to 
computing an entire approximation.
"""
function at_other_methods_rs!(at)
  at_in = at.at_in
  piv = at_in.pcmc.piv
  isMJP = at_in.pcmc.mc isa MarkovJumpProcess
  if isMJP
    Q = at_in.pcmc.mc.Q
  elseif at_in.pcmc.mc isa MarkovChain
    P = at_in.pcmc.mc.P
    Q = P-I
  else 
    throw(ArgumentError("Not MJP or MC."))
  end

  for (rname,rval) in at_in.rs
    for (i1, t1) in enumerate(rval.tvals)
      println("Other Methods (r): r=",rname,", t1=", t1)
      Aidx = get_linear_sublevel_set(at_in.pcmc.mc.vss, t1)
      NrIdx = get_strict_interior(at_in.pcmc.mc, t1)
      NrIdx_A = findall(x->x in NrIdx, Aidx)
      Q_Aidx = Q[Aidx, Aidx]

      # RTA
      if rval.algflags.rta 
        if isMJP
          minvec, maxvec, avgvec = rta_Q(Q, Aidx)
        else
          minvec, maxvec, avgvec = rta(P, Aidx)
        end

        minvec_S = Aidx_to_Sidx(at.at_in.pcmc.mc.vss, minvec, Aidx)
        maxvec_S = Aidx_to_Sidx(at.at_in.pcmc.mc.vss, maxvec, Aidx)
        avgvec_S = Aidx_to_Sidx(at.at_in.pcmc.mc.vss, avgvec, Aidx)

        at.rta.rs[rname].conditional_lbs[i1] = minvec_S'*rval.r_S
        at.rta.rs[rname].conditional_ubs[i1] = maxvec_S'*rval.r_S
        

      end


      for (in, rta_lb, rta_ub, lpoa_lb, lpoa_ub) in zip(rval.momentbounds,
                                      at.rta.rs[rname].lbs.mb,
                                      at.rta.rs[rname].ubs.mb,
                                      at.lpoa.rs[rname].lbs.mb,
                                      at.lpoa.rs[rname].ubs.mb)
        if in.active
          tk = in.tvals[i1]+1
          c = in.bound

          #RTA
          if rval.algflags.rta
            r_S = rval.r_S
            r_A = r_S[Aidx]
            # lower bound only holds for r non-negative.
            at.rta.rs[rname].approx_errs[i1] = (avgvec'* r_A)[1] 
            rta_lb[i1] = ((1-c/tk)*minvec'* r_A)[1] # (4.38) in [1]
            rta_ub[i1] = (maxvec'* r_A)[1]+c*rval.r_over_w[i1] # (4.40) in [1]
          end

          if rval.algflags.lpoa
            w_S = in.w_S
            w_Aidx = w_S[Aidx]
            piv_lpoa_lb = LPOA_Bound(Q_Aidx, NrIdx_A, c, tk, w_Aidx, r_A, upper=false)
            lpoa_lb[i1] = (piv_lpoa_lb'*r_A)[1]
            piv_lpoa_ub = LPOA_Bound(Q_Aidx, NrIdx_A, c, tk, w_Aidx, r_A, upper=true)
            at.lpoa.rs[rname].ubs_r[i1] = (piv_lpoa_ub'*r_A)[1]
            lpoa_ub[i1] = at.lpoa.rs[rname].ubs_r[i1]+c*rval.r_over_w[i1]

            if at.at_in.benchmark
              b = @benchmark LPOA_Bound($Q_Aidx, $NrIdx_A, $c, $tk, $w_Aidx, $r_A, upper=false) 
              at.lpoa.rs[rname].times[i1] = median(b.times)
              at.lpoa.rs[rname].memory[i1] = b.memory
            end
          end
        end
      end
      if at.at_in.tailmass_bound
        tk = at.at_in.tvals[i1]+1
        c = at.at_in.tailmass_vec[i1]*tk

        #RTA
        r_S = at.rinfos[rname].r_S
        r_A = r_S[Aidx]
        # these lower and upper bounds only hold for r non-negative.
        at.rta.rs[rname].approx_errs[i1] = (avgvec'* r_A)[1]
        at.rta.rs[rname].lbs.tb[i1] = ((1-c/tk)*minvec'* r_A)[1]
        at.rta.rs[rname].ubs.tb[i1] = (maxvec'* r_A)[1]+c*rval.r_over_w[i1]

        if rval.algflags.lpoa
          w_S = in.w_S
          w_Aidx = w_S[Aidx]
          piv_lpoa_lb = LPOA_Bound(Q_Aidx, NrIdx_A, c, tk, w_Aidx, r_A, upper=false)
          at.lpoa.rs[rname].lbs.tb[i1] = (piv_lpoa_lb'*r_A)[1] # (4.38) in [1]
          piv_lpoa_ub = LPOA_Bound(Q_Aidx, NrIdx_A, c, tk, w_Aidx, r_A, upper=true) 
          at.lpoa.rs[rname].ubs.tb[i1] = (piv_lpoa_ub'*r_A)[1]+c*rval.r_over_w[i1] # (4.40) in [1]

          if at.at_in.benchmark
            b = @benchmark LPOA_Bound($Q_Aidx, $NrIdx_A, $c, $tk, $w_Aidx, $r_A, upper=false) 
            at.lpoa.rs[rname].times[i1] = median(b.times)
            at.lpoa.rs[rname].memory[i1] = b.memory
          end
        end
      end
    end
  end
end


"""
    at_fixedKruns!(at)

This method runs tests on the methods presented in [2]. Here, the set ``K`` is 
fixed, and we vary ``A``.
"""
function at_fixedKruns!(at)
  pcmc=at.at_in.pcmc
  Sidx = 1:pcmc.mc.vss.size
  at_in = at.at_in

  isMJP = at_in.pcmc.mc isa MarkovJumpProcess
  if isMJP
    Q = at_in.pcmc.mc.Q
    P, d = get_R_matrix(Q)
  elseif at_in.pcmc.mc isa MarkovChain
    P = at_in.pcmc.mc.P
  else 
    throw(ArgumentError("Not MJP or MC."))
  end 
  
  for (rname, rval) in at.at_in.rs
    if at.at_in.pcmc.mc isa MarkovJumpProcess
      r1_S = (1.0 ./d).*rval.r_S
      r2_S = (1.0 ./d).*ones(at.at_in.pcmc.mc.vss.size)
    elseif at.at_in.pcmc.mc isa MarkovChain
      r1_S = rval.r_S  
      r2_S = ones(at.at_in.pcmc.mc.vss.size)
    else 
      throw(ArgumentError("Not MC or MJP."))          
    end
    println("FixedK run: rname=", rname)
    for (fk_name,fk_in) in rval.fk_ins
      for (i,t) in enumerate(rval.tvals)
        println("FixedK run: t1=", t)
        Aidx = get_linear_sublevel_set(at.at_in.pcmc.mc.vss, t)
        if issubset(fk_in.Kidx, Aidx)      
          Acidx = setdiff(Sidx, Aidx)

          h1_S = optimal_h(P, fk_in.f1_S, Acidx)
          h2_S = optimal_h(P, fk_in.f2_S, Acidx)
          apx_out = approx(fk_in.Kidx, Aidx, P, row_normalize)
          CS_weighted_apx_out = CS_ImG_weighted_approx(fk_in.Kidx, Aidx, P)
          CS_min_tv_apx_out = CS_min_tv_approx(fk_in.Kidx, Aidx, P)
          pf_apx_out = approx(fk_in.Kidx, Aidx, P, pf_normalize)

          if isMJP
            apx_piv_S = rescale_piv(apx_out.piv_S,1.0 ./d)
            CS_weighted_apx_piv_S = rescale_piv(CS_weighted_apx_out.piv_S,1.0 ./d)
            CS_min_tv_apx_piv_S = rescale_piv(CS_min_tv_apx_out.piv_S,1.0 ./d)
            pf_apx_out_piv_S = rescale_piv(pf_apx_out.piv_S, 1.0 ./d)
          else
            apx_piv_S = apx_out.piv_S
            CS_weighted_apx_piv_S = CS_weighted_apx_out.piv_S
            CS_min_tv_apx_piv_S = CS_min_tv_apx_out.piv_S
            pf_apx_out_piv_S =pf_apx_out.piv_S
          end      
          
          at.IO.fixedK[fk_name].apx_out.rn.apx_errs[i] = tv(pcmc.piv, apx_piv_S)
          at.IO.fixedK[fk_name].apx_out.CS_weighted.apx_errs[i] = tv(pcmc.piv, CS_weighted_apx_piv_S)
          at.IO.fixedK[fk_name].apx_out.CS_min_tv.apx_errs[i] = tv(pcmc.piv, CS_min_tv_apx_piv_S)
          at.IO.fixedK[fk_name].apx_out.pf.apx_errs[i]= tv(pcmc.piv, pf_apx_out_piv_S)


          b = @benchmark approx($fk_in.Kidx, $Aidx, $P, $row_normalize)
          at.IO.fixedK[fk_name].apx_out.rn.apx_times[i] = median(b.times)

          b = @benchmark CS_ImG_weighted_approx($fk_in.Kidx, $Aidx, $P)
          at.IO.fixedK[fk_name].apx_out.CS_weighted.apx_times[i] = median(b.times)

          b = @benchmark CS_min_tv_approx($fk_in.Kidx, $Aidx, $P)
          at.IO.fixedK[fk_name].apx_out.CS_min_tv.apx_times[i] = median(b.times)

          b = @benchmark approx($fk_in.Kidx, $Aidx, $P, $pf_normalize)
          at.IO.fixedK[fk_name].apx_out.pf.apx_times[i] = median(b.times)

          if rval.algflags.lp
            println("LP bounds: ")

            lp_out = lp_bounds(fk_in.Kidx, Aidx, P, r1_S, h1_S, h2_S, e_S=r2_S)
            at.IO.fixedK[fk_name].rout[rname].lp_lb[i] = lp_out.lb
            at.IO.fixedK[fk_name].rout[rname].lp_ub[i] = lp_out.ub


            b = @benchmark lp_bounds($fk_in.Kidx, $Aidx, $P,
                                     $r1_S, $h1_S,
                                     $h2_S) 

            at.IO.fixedK[fk_name].rout[rname].lp_times[i] = median(b.times)
          end

          if rval.algflags.ptb

            # Since ptb method is slightly faster knowing r2=e, putting an if-statement here.
            if isMJP
              rn_out = ptb_bounds(fk_in.Kidx, Aidx, P,
                                  r1_S, h1_S,
                                  r2_S, h2_S, 
                                  row_normalize)
            else
              rn_out = ptb_bounds(fk_in.Kidx, Aidx, P,
                                  r1_S, h1_S,
                                  h2_S, 
                                  row_normalize)
            end

            at.IO.fixedK[fk_name].rout[rname].rn.ptb_lbs[i] = rn_out.lb
            at.IO.fixedK[fk_name].rout[rname].rn.ptb_ubs[i] = rn_out.ub
            at.IO.fixedK[fk_name].rout[rname].rn.rweightedTVs[i] = rn_out.tvb

            if isMJP
              b = @benchmark ptb_bounds($fk_in.Kidx, $Aidx, $P,
                $r1_S, $h1_S,
                $r2_S, $h2_S, 
                $row_normalize)
            else
              b = @benchmark ptb_bounds($fk_in.Kidx, $Aidx, $P,
                $r1_S, $h1_S,
                $h2_S, $row_normalize)

            end
            at.IO.fixedK[fk_name].rout[rname].rn.ptb_bds_times[i] = median(b.times)
          end
          if rval.algflags.CS
            if isMJP
              CS_out = CS_r_bounds(fk_in.Kidx, Aidx, P,
                                  r1_S, h1_S,
                                  h2_S, e_S=r2_S)
              b = @benchmark CS_r_bounds($fk_in.Kidx, $Aidx, $P,
                                        $r1_S, $h1_S,
                                        $h2_S, e_S=$r2_S)

              CS_out_heuristic = CS_r_bounds_stable_max_row_sum_heuristic(fk_in.Kidx, Aidx, P,
                                              r1_S, h1_S,
                                              h2_S, e_S=r2_S)
              b_heuristic = @benchmark CS_r_bounds_stable_max_row_sum_heuristic($fk_in.Kidx, $Aidx, $P,
                                              $r1_S, $h1_S,
                                              $h2_S, e_S=$r2_S)
              
                                        
              CS_tv_out = CS_tv_bounds(fk_in.Kidx, Aidx, P, r1_S, h1_S, h2_S, row_normalize, e_S=r2_S)
              b2 = @benchmark CS_tv_bounds($fk_in.Kidx, $Aidx, $P, $r1_S, $h1_S, $h2_S, $row_normalize, e_S=$r2_S)


            else
              CS_out = CS_r_bounds(fk_in.Kidx, Aidx, P,
                                  r1_S, h1_S,
                                  h2_S)
              b = @benchmark CS_r_bounds($fk_in.Kidx, $Aidx, $P,
                                        $r1_S, $h1_S,
                                        $h2_S)

              CS_out_heuristic = CS_r_bounds_stable_max_row_sum_heuristic(fk_in.Kidx, Aidx, P,
                                              r1_S, h1_S,
                                              h2_S)
              b_heuristic = @benchmark CS_r_bounds_stable_max_row_sum_heuristic($fk_in.Kidx, $Aidx, $P,
                                              $r1_S, $h1_S,
                                              $h2_S)

              CS_tv_out = CS_tv_bounds(fk_in.Kidx, Aidx, P, r1_S, h1_S, h2_S, row_normalize)
              b2 = @benchmark CS_tv_bounds($fk_in.Kidx, $Aidx, $P, $r1_S, $h1_S, $h2_S, $row_normalize)
            end
            at.IO.fixedK[fk_name].rout[rname].CS_lb[i] = CS_out.lb
            at.IO.fixedK[fk_name].rout[rname].CS_ub[i] = CS_out.ub
            at.IO.fixedK[fk_name].rout[rname].CS_times[i] = median(b.times)

            at.IO.fixedK[fk_name].rout[rname].CS_heuristic_lb[i] = CS_out_heuristic.lb
            at.IO.fixedK[fk_name].rout[rname].CS_heuristic_ub[i] = CS_out_heuristic.ub
            at.IO.fixedK[fk_name].rout[rname].CS_heuristic_times[i] = median(b_heuristic.times)

            at.IO.fixedK[fk_name].rout[rname].rn.CS_ptb_lbs[i] = CS_tv_out.lb
            at.IO.fixedK[fk_name].rout[rname].rn.CS_ptb_ubs[i] = CS_tv_out.ub
            at.IO.fixedK[fk_name].rout[rname].rn.CS_rweightedTVs[i] = CS_tv_out.tvb
            at.IO.fixedK[fk_name].rout[rname].rn.CS_rweightedTVstimes[i] = median(b2.times)
          end

        end
      end
    end
  end
end


# # The code below is currently not in use, but includes code for potential future
# # testing.

# function at_Regenerative!(at)
#   pcmc=at.at_in.pcmc
#   Sidx = 1:pcmc.mc.vss.size
#   at_in = at.at_in

#   isMJP = at_in.pcmc.mc isa MarkovJumpProcess
#   if isMJP
#     Q = at_in.pcmc.mc.Q
#     P, d = get_R_matrix(Q)
#   elseif at_in.pcmc.mc isa MarkovChain
#     P = at_in.pcmc.mc.P
#   else 
#     throw(ArgumentError("Not MJP or MC."))
#   end 
  
#   for (rname, rval) in at.at_in.rs
#     if at.at_in.pcmc.mc isa MarkovJumpProcess
#       r1_S = (1.0 ./d).*rval.r_S
#       r2_S = (1.0 ./d).*ones(at.at_in.pcmc.mc.vss.size)
#     elseif at.at_in.pcmc.mc isa MarkovChain
#       r1_S = rval.r_S  
#       r2_S = ones(at.at_in.pcmc.mc.vss.size)
#     else 
#       throw(ArgumentError("Not MC or MJP."))          
#     end
#     println("FixedK run: rname=", rname)
#     for (fk_name,fk_in) in rval.fk_ins
#       for (i,t) in enumerate(rval.tvals)
#         println("FixedK run: t1=", t)
#         Aidx = get_linear_sublevel_set(at.at_in.pcmc.mc.vss, t)
#         if issubset(fk_in.Kidx, Aidx)      
#           Acidx = setdiff(Sidx, Aidx)
#           h1_S = optimal_h(P, fk_in.f1_S, Acidx)
#           h2_S = optimal_h(P, fk_in.f2_S, Acidx)
#           apx_out = approx(fk_in.Kidx, Aidx, P, row_normalize)
#           CS_weighted_apx_out = CS_ImG_weighted_approx(fk_in.Kidx, Aidx, P)
#           CS_min_tv_apx_out = CS_min_tv_approx(fk_in.Kidx, Aidx, P)

#           if isMJP #isMarkovJumpProcess?
#             # compute your approximation 
#             # (may need to be reweighted if Markov Jump Process)
#             # your approx =  regen_approx(...)
#             # b_regen_approx = @benchmark regen_approx($...) #use dollar signs before every variable when timing code (benchmark)
#           else
#             # compute your approximation 
#           end      

#           at.regenerative.fixedK[fk_name].regenerative_apx[i].apx_errs = 0.0 #tv(pcmc.piv, your approximation)
#           at.regenerative.fixedK[fk_name].regenerative_apx[i].apx_times = 0.0 #median(b_regen_approx.times)

#           if rval.algflags.regenerative
#             # regen_out_eq_exp = # compute eq exp bounds
#             # b_regen_eq_exp = @benchmark regen($ )

#             # regen_out_tv # compute tv bound
#             # b_regen_eq_exp = @benchmark regen($ )



#             at.regenerative.fixedK[fk_name].rout[rname].lbs = 0.0
#             at.regenerative.fixedK[fk_name].rout[rname].ubs = 0.0

#             at.regenerative.fixedK[fk_name].rout[rname].lbs = 0.0 #regen_out_eq_exp.lb
#             at.regenerative.fixedK[fk_name].rout[rname].ubs = 0.0 #regen_out_eq_exp.ub
#             at.regenerative.fixedK[fk_name].rout[rname].ubs = 0.0 #regen_out_eq_exp.ub

#             at.regenerative.fixedK[fk_name].rout[rname].rweighted_tvs_bds = 0.0 
#             at.regenerative.fixedK[fk_name].rout[rname].rweighted_tvs_bds_times = 0.0 # median(b_regen_ptb_tim)



            

            
#           end      

#         end
#       end
#     end
#   end
# end

# """
#     at_make_data!(at)

# Currently not in use. For potential future experiments that include both 
# running over different choices of ``K`` and ``A``.
# """
# function at_make_data!(at)

#   if at.at_in.run_levelsets
#     at_inner_outer_level!(at)
#     save(string("Algorithms/Truncation/Tests/AllTests/Data/",
#       at.at_in.pcmc.name, "-comparisons.jld2"),
#       Dict("at" => at))

#     at_inner_outer_sublevel!(at)
#     save(string("Algorithms/Truncation/Tests/AllTests/Data/",
#     at.at_in.pcmc.name, "-comparisons.jld2"),
#     Dict("at" => at))
#   end

#   at_fixedKruns!(at)
  
#   save(string("Algorithms/Truncation/Tests/AllTests/Data/",
#     at.at_in.pcmc.name, "-comparisons.jld2"),
#     Dict("at" => at))

#   at_other_methods_dist!(at)

#   save(string("Algorithms/Truncation/Tests/AllTests/Data/",
#   at.at_in.pcmc.name, "-comparisons.jld2"),
#   Dict("at" => at))
#   at_other_methods_rs!(at)

#   save(string("Algorithms/Truncation/Tests/AllTests/Data/",
#     at.at_in.pcmc.name, "-comparisons.jld2"),
#     Dict("at" => at))
#   return
# end

# function at_rinfos!(at)
#  # Lyapunov Stuff
#   at_in = at.at_in
#   for (rname, rval) in at_in.rs
#     at.rinfos[rname] = (LyapunovOut= get_all_K(at_in.pcmc.mc, rval.r_S, rval.f1_S, ones(at_in.pcmc.mc.vss.size), f2, at_in.tvals, vss_pow_function(at_in.pcmc.mc.vss, 1)),
#                      r_S = at_in.rs_S[i])
#   end
# end

# function at_inner_outer_level!(at)

#   at_in = at.at_in
#   piv = at_in.pcmc.piv
#   P = at_in.pcmc.mc.P



#   for (i1,t1) in enumerate(at_in.tvals)
#     Aidx = get_linear_sublevel_set(at_in.pcmc.mc.vss, t1)
#     for (i2,t2) in enumerate(at_in.tvals)
#       if t2 > t1
#         continue
#       end

#       println("Level: t1=", t1, ", t2=", t2)

#       Kidx = get_linear_level_set(at_in.pcmc.mc.vss, t2)

#       # "Firstflag" We save the quantities that are not specific to an r after the first run
#       # Maybe in the future write a version of the "saveall" code that does not take 
#       # in an r. But the thing is that currently the saving of F_i and so on happens
#       # as part of the bounds code, which depends on an r. 

#       firstflag = true  

#       for (rname,rval) in at_in.rs 
#         out = full_alg_save_quantities(Kidx, Aidx, P, rval.r_S)

#         # save quantities that are not r-dependent
#         if firstflag
#           # General Quantities
#           at.IO.level.condImP22s[i1, i2] = out.ioapx.condImP22
#           at.IO.level.rho_Gs[i1, i2] = out.ioapx.rho_G
#           at.IO.level.minrowsum_Gs[i1, i2] = out.ioapx.minrowsum_G
    
#           # APX DATA
#           # Row Normalization 
#           at.IO.level.apx_out.rn.apx_errs[i1, i2] = tvzp(out.ioapx.rn.piv_S, piv)
#           # PF
#           at.IO.level.apx_out.pf.apx_errs[i1, i2] = tvzp(out.ioapx.pf.piv_S, piv)
#           #cond
#           at.IO.level.apx_out.cond.apx_errs[i1,i2] = tvzp(out.ioapx.cond.piv_S, piv)
#           #pi3
#           at.IO.level.apx_out.pi3.apx_errs[i1,i2] = tvzp(out.ioapx.pi3.piv_S, piv)

#           if at.at_in.benchmark
#             b = @benchmark approx($Kidx, $Aidx, $P, $row_normalize) 
#             at.IO.level.apx_out.rn.apx_times[i1,i2] = median(b.times)
#             at.IO.level.apx_out.rn.apx_memory[i1,i2] = b.memory


#             # b = @benchmark approx($Kidx, $Aidx, $P, $pf_normalize) 
#             # at.IO.level.apx_out.pf.apx_times[i1,i2] = median(b.times)
#             # at.IO.level.apx_out.pf.apx_memory[i1,i2] = b.memory


#             # b = @benchmark pi3_approx($Kidx, $Aidx, $P) 
#             # at.IO.level.apx_out.pi3.apx_times[i1,i2] = median(b.times)
#             # at.IO.level.apx_out.pi3.apx_memory[i1,i2] = b.memory
#           end

#           firstflag = false
#         end
#           r_S = rval.r_S
#           at.IO.level.routs[rname].r_apx_out.rn.r_apx_values[i1,i2] = out.ioapx.rn.piv_S'*r_S
#           at.IO.level.routs[rname].r_apx_out.pf.r_apx_values[i1,i2] = out.ioapx.pf.piv_S'*r_S
#           at.IO.level.routs[rname].r_apx_out.cond.r_apx_values[i1,i2] = out.ioapx.cond.piv_S'*r_S
#           at.IO.level.routs[rname].r_apx_out.pi3.r_apx_values[i1,i2] = out.ioapx.pi3.piv_S'*r_S
#           # at.IO.level.routs[rname].r_apx_out.exit.r_apx_values[i1,i2] = out.ioapx.exit.piv_S'*r_S
#       end          
#     end
#     save(string("Algorithms/Truncation/Tests/AllTests/Data/",
#       at.at_in.pcmc.name, "-comparisons.jld2"),
#       Dict("at" => at))
#   end
#   return
# end

# function at_inner_outer_sublevel!(at)
  
#   at_in = at.at_in
#   piv = at_in.pcmc.piv
#   P = at_in.pcmc.mc.P
#   Sidx = collect(1:at_in.pcmc.mc.vss.size)

#   # Gurobi_ENV = Gurobi.Env()

#   for (i1,t1) in enumerate(at_in.tvals)
#     Aidx = get_linear_sublevel_set(at_in.pcmc.mc.vss, t1)
#     for (i2,t2) in enumerate(at_in.tvals)
#       if t2 > t1
#         continue
#       end

#       println("Sublevel: t1=", t1, ", t2=", t2) 
#       Kidx = get_linear_sublevel_set(at_in.pcmc.mc.vss, t2)

#       # "Firstflag" We save the quantities that are not specific to an r after the first run
#       # Maybe in the future write a version of the "saveall" code that does not take 
#       # in an r. But the thing is that currently the saving of F_i and so on happens
#       # as part of the bounds code, which depends on an r. 

#       firstflag = true  

#       for (rname,rval) in at_in.rs
#         lo = rval.Lyapunov_out
#         do_bounds_r  = i2>=lo.tminidx
#         if do_bounds_r
#           println("Doing Full Algorithm Save Quantities (Bounds)")
#           g1_S = lo.c1vec[i2-lo.tminidx+1]*lo.f1_S
#           g2_S = lo.c2vec[i2-lo.tminidx+1]*lo.f2_S
#           h1_S = optimal_h(at_in.pcmc.mc, g1_S, setdiff(Sidx, Aidx))
#           h2_S = optimal_h(at_in.pcmc.mc, g2_S, setdiff(Sidx, Aidx))
#           out = full_alg_save_quantities(Kidx, Aidx, P, lo.r1_S, 
#                             h1_S, h2_S, PF_tol = at_in.pcmc.err)
#           println("Finished Full Algorithm Save Quantities (Bounds)")
#         else
#           println("Doing Full Algorithm Save Quantities (No bounds)")
#           out = full_alg_save_quantities(Kidx, Aidx, P, rval.r_S)
#           println("Finished Full Algorithm Save Quantities (No bounds)")
#         end

#         # save quantities that are not r-dependent
#         if firstflag
#           # General Quantities
#           at.IO.sublevel.condImP22s[i1, i2] = out.ioapx.condImP22
#           at.IO.sublevel.rho_Gs[i1, i2] = out.ioapx.rho_G
#           at.IO.sublevel.minrowsum_Gs[i1,i2] = out.ioapx.minrowsum_G
          
#           # APX DATA
#           # Row Normalization 
#           at.IO.sublevel.apx_out.rn.apx_errs[i1, i2] = tvzp(out.ioapx.rn.piv_S, piv)
#           # PF
#           at.IO.sublevel.apx_out.pf.apx_errs[i1, i2] = tvzp(out.ioapx.pf.piv_S, piv)
#           #cond
#           at.IO.sublevel.apx_out.cond.apx_errs[i1,i2] = tvzp(out.ioapx.cond.piv_S, piv)
#           #pi3
#           at.IO.sublevel.apx_out.pi3.apx_errs[i1,i2] = tvzp(out.ioapx.pi3.piv_S, piv)


#           if at.at_in.benchmark
#             println("Benchmarking Approximations")
#             b = @benchmark approx($Kidx, $Aidx, $P, $row_normalize)  
#             at.IO.sublevel.apx_out.rn.apx_times[i1,i2] = median(b.times)
#             at.IO.sublevel.apx_out.rn.apx_memory[i1,i2] = b.memory



#             # b = @benchmark approx($Kidx, $Aidx, $P, $pf_normalize)  
#             # at.IO.sublevel.apx_out.pf.apx_times[i1,i2] = median(b.times)
#             # at.IO.sublevel.apx_out.pf.apx_memory[i1,i2] = b.memory


#             # b = @benchmark pi3_approx($Kidx, $Aidx, $P)  
#             # at.IO.sublevel.apx_out.pi3.apx_times[i1,i2] = median(b.times)
#             # at.IO.sublevel.apx_out.pi3.apx_memory[i1,i2] = b.memory
#             # println("Finished Benchmarking Approximations")
#           end
#           firstflag = false
#         end

#         r_S = rval.r_S
#         at.IO.sublevel.routs[rname].r_apx_out.rn.r_apx_values[i1,i2] = out.ioapx.rn.piv_S'*r_S
#         at.IO.sublevel.routs[rname].r_apx_out.pf.r_apx_values[i1,i2] = out.ioapx.pf.piv_S'*r_S
#         at.IO.sublevel.routs[rname].r_apx_out.cond.r_apx_values[i1,i2] = out.ioapx.cond.piv_S'*r_S
#         at.IO.sublevel.routs[rname].r_apx_out.pi3.r_apx_values[i1,i2] = out.ioapx.pi3.piv_S'*r_S

#         if do_bounds_r

#           at.IO.sublevel.routs[rname].rn.rweightedTVs[i1,i2] = out.iobds.rn.ptb_tv
#           at.IO.sublevel.routs[rname].rn.ptb_lbs[i1,i2] = out.iobds.rn.ptb_lb
#           at.IO.sublevel.routs[rname].rn.ptb_ubs[i1,i2] = out.iobds.rn.ptb_ub

#           at.IO.sublevel.routs[rname].rn.cond_Finvs[i1,i2] = out.iobds.rn.cond_Finv
#           at.IO.sublevel.routs[rname].rn.Deltas[i1,i2] = out.iobds.rn.Delta
#           at.IO.sublevel.routs[rname].rn.minrowsum_Finvs[i1,i2] = out.iobds.rn.minrowsum_Finv
#           at.IO.sublevel.routs[rname].rn.maxrowsum_Finvs[i1,i2] = out.iobds.rn.maxrowsum_Finv

#           at.IO.sublevel.routs[rname].pf.rweightedTVs[i1,i2] = out.iobds.rn.ptb_tv
#           at.IO.sublevel.routs[rname].pf.ptb_lbs[i1,i2] = out.iobds.rn.ptb_lb
#           at.IO.sublevel.routs[rname].pf.ptb_ubs[i1,i2] = out.iobds.rn.ptb_ub

#           at.IO.sublevel.routs[rname].pf.cond_Finvs[i1,i2] = out.iobds.rn.cond_Finv
#           at.IO.sublevel.routs[rname].pf.Deltas[i1,i2] = out.iobds.rn.Delta
#           at.IO.sublevel.routs[rname].pf.minrowsum_Finvs[i1,i2] = out.iobds.rn.minrowsum_Finv
#           at.IO.sublevel.routs[rname].pf.maxrowsum_Finvs[i1,i2] = out.iobds.rn.maxrowsum_Finv    
          
#           at.IO.sublevel.routs[rname].lp_lb[i1,i2] = out.iobds.lp_lb
#           at.IO.sublevel.routs[rname].lp_ub[i1,i2] = out.iobds.lp_ub   

#           # r1_S  = lo.r1_S
#           # g1_S = lo.c1vec[i2-lo.tminidx+1]*lo.f1_S
#           # g2_S = lo.c2vec[i2-lo.tminidx+1]*lo.f2_S

#           # h1_S = optimal_h(pcmc.mc, g1_S, setdiff(collect(1:size(P,1)),Aidx))
#           # h2_S = optimal_h(pcmc.mc, g2_S, setdiff(collect(1:size(P,1)),Aidx))

#           if at.at_in.benchmark
#             if rval.algflags.ptb
#               println("Benchmarking PTB Bounds (Row)")

#               b = @benchmark ptb_bounds($Kidx, $Aidx, $P, $r_S,
#                                                                 $h1_S, 
#                                                                 $h2_S, 
#                                                                 $row_normalize)  
#               at.IO.sublevel.routs[rname].rn.ptb_bds_times[i1,i2] = median(b.times)
#               at.IO.sublevel.routs[rname].rn.ptb_bds_memory[i1,i2] = b.memory
#               println("Finished Benchmarking PTB Bounds (Row):   ", rname)
#             end

#             # println("Benchmarking PTB Bounds (PF)")
#             # # b = @benchmarkable ptb_bounds($Kidx, $Aidx, $P, $r1_S, 
#             # #                                                   $h1_S, 
#             # #                                                   $h2_S, 
#             # #                                                   $pf_normalize)  
#             # # b2 = run(b)
#             # b2 = @benchmark ptb_bounds($Kidx, $Aidx, $P, $r_S, 
#             #                                             $h1_S, 
#             #                                             $h2_S, 
#             #                                             $pf_normalize) 
#             # at.IO.sublevel.routs[rname].pf.ptb_bds_times[i1,i2] = median(b2.times)
#             # at.IO.sublevel.routs[rname].pf.ptb_bds_memory[i1,i2] = b2.memory
#             # println("Finished Benchmarking PTB Bounds (PF)")
#             if rval.algflags.lp
#               println("Benchmarking LP Bounds:   ", rname)

#               b = @benchmark lp_bounds($Kidx, $Aidx, $P, $r_S, 
#                                                       $h1_S, 
#                                                       $h2_S
#                                                       )

#               at.IO.sublevel.routs[rname].lp_times[i1,i2] = median(b.times)
#               at.IO.sublevel.routs[rname].lp_memory[i1,i2] = b.memory
#               println("Finished Benchmarking LP Bounds")
#             end
#           end
#         end   
#       end          
#     end
#     save(string("Algorithms/Truncation/Tests/AllTests/Data/",
#       at.at_in.pcmc.name, "-comparisons.jld2"),
#       Dict("at" => at))
#   end
#   return
# end

#= References
[1] Kuntz, Juan, et al. "Stationary distributions of continuous-time Markov chains: a review of theory and truncation-based approximations." SIAM Review 63.1 (2021): 3-64.
[2] Infanger, Alex and Glynn, Peter W. "A New Truncation Algorithm for Markov Chain Equilibrium Distributions with Computable Error Bounds".
=# 

