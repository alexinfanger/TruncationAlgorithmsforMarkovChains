vss = VectorStateSpace([300,300])
tsp = ToggleSwitchParams((90.0, 1.0,vss))
Q = get_Q_TS(tsp)
piv_reflect = get_stat_dist_Q(Q)
R,d = get_R_matrix(Q)
Sidx = collect(1:vss.size)

pi_mode = ind_to_vec(vss,findmax(piv_reflect)[2]) 
pi_mode_2 = [10,10]
if pi_mode != pi_mode_2
      throw(ErrorException("Error in mode calculation."))
end

e_S = vss_func_to_vec(vss, x->1);
lin_S = vss_abspow_vec(vss, 1);
lin_c_S = vss_centered_abspow_vec(vss,1, pi_mode);
sq_S = vss_abspow_vec(vss, 2);
sq_c_S = vss_centered_abspow_vec(vss,2, pi_mode);

tilde_e_S = Vector(e_S./d)
tilde_lin_S = Vector(lin_S./d)

lo_e = get_K_rate_matrix_simple(Q, e_S, lin_c_S)
K_e = lo_e.K
K_union = get_K_union(Q, e_S, lin_c_S, lin_S, sq_c_S)


t = 200
Aidx = get_linear_sublevel_set(vss, t)
h1_S = optimal_h(R, sq_c_S, setdiff(Sidx, Aidx))
h2_S = optimal_h(R, lin_c_S, setdiff(Sidx, Aidx))

rn_out = approx(K_e, Aidx, R, row_normalize)
piv = rn_out.piv_S./d
piv = Vector(piv./sum(piv))


# Confirming the solution solution computed agrees with truncation and reflection
# as a check.
println(tv(piv,piv_reflect))

out_e = CS_tv_bounds(K_e, Aidx, R, tilde_e_S, h2_S, h2_S, row_normalize, e_S = tilde_e_S)
out_lin = CS_tv_bounds(K_union, Aidx, R, tilde_lin_S, h2_S, h1_S, row_normalize, e_S = tilde_e_S)

mc = ReflectedToggleSwitch(tsp.br, tsp.dr, vss, Q)

eq_exp_e = PrecomputedEqExp((out_e, "one"))
eq_exp_lin =  PrecomputedEqExp((out_lin, "lin"))


pcmc = PrecomputedToggleSwitch(mc, 
        piv, out_e.tvb, "TS-90-1", 
        Dict(eq_exp_e.name=>eq_exp_e, 
                 eq_exp_lin.name=>eq_exp_lin))

save("src/models/toggle_switch/data/TS-90-1.jld2",
      Dict("pcmc"=>pcmc))


