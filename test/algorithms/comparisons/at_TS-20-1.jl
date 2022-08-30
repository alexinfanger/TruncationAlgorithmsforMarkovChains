pcmc = (load("src/models/toggle_switch/data/TS-20-1.jld2"))["pcmc"]
vss = pcmc.mc.vss
Q = pcmc.mc.Q
Sidx = collect(1:vss.size)
pi_mode = ind_to_vec(vss,findmax(pcmc.piv)[2])

e_S = vss_func_to_vec(vss, x->1);
lin_S = vss_abspow_vec(vss, 1);
lin_c_S = vss_centered_abspow_vec(vss,1, pi_mode);
sq_S = vss_abspow_vec(vss, 2);
sq_c_S = vss_centered_abspow_vec(vss,2, pi_mode);
cube_S = vss_abspow_vec(vss, 3);
cube_c_S = vss_centered_abspow_vec(vss, 3, pi_mode);
cube_sp_S = vss_sumpow_vec(vss, 3)
w_S = vss_abspow_vec(vss, 6);

# Confirm the Lyapunov analysis in [1], putting a theoretical apriori upper bound
# on how large of a truncation set we have to look at to determine K.
c0 = 2*pcmc.mc.br
c1 = 1+2*pcmc.mc.br+2*pcmc.mc.dr*(2*(pi_mode[1]-1)+1)
c2 = 2*pcmc.mc.dr

n_1 = ceil((c1 + sqrt(c1^2+(4*c2*c0)/2))/c2)
n_2 = ceil((2*pcmc.mc.br+4*pcmc.mc.dr*(pi_mode[1]-1)+1) /pcmc.mc.dr)
n_union = max(n_1,n_2)


A_e = get_linear_sublevel_set(vss, n_1)
K_e = get_K_rate_matrix_simple_on_subset(Q, e_S, lin_c_S, A_e)
b_e = @benchmark get_K_rate_matrix_simple_on_subset($Q, $e_S, $lin_c_S, $A_e)
println("Correct Lyapunov function and K pair (r(x)=1):  ", is_Lyapunov_function_rate(Q, setdiff(Sidx, K_e), e_S, lin_c_S))

A_union = get_linear_sublevel_set(vss, max(n_1,n_2))
K_union = get_K_union_on_subset(Q, e_S, lin_c_S, lin_S, sq_c_S, A_union)
b_union = @benchmark get_K_union_on_subset($Q, $e_S, $lin_c_S, $lin_S, $sq_c_S, $A_union)

println("Correct Lyapunov function and K pair (r(x)=1 and r(x)=x_1+x_2):  ", is_Lyapunov_function_rate(Q, setdiff(Sidx, K_union), lin_S, sq_c_S), ", ",is_Lyapunov_function_rate(Q, setdiff(Sidx, K_union), e_S, lin_c_S))


r0tvals=[30,40,50,60,70]

fake_multiK_out = LyapunovMultiKOut(
  [],
  [],
  [],
  [],
  [],
  [],
  [],
  [],
  1,
  1,
  [],
  []	
	)

fk_in = (tvals =  r0tvals,
				 Kidx = K_e,
				 r_S = e_S, 
				 f1_S = lin_c_S, 
				 f2_S = lin_c_S,
				 name="Pg")


SDP = (active = false, bound = 1.8e7, tvals=r0tvals, w_S = w_S, name="2-norm-square")
Lyapunov = (active=false, bound=0.0,  tvals=r0tvals, w_S = [0.0], name="N/A")
exact = (active= false, bound = 0.0, tvals=r0tvals, w_S = [0.0], name="N/A")
mb = MomentBounds((SDP, Lyapunov, exact))

ATr0 = ATRewardFunction((
	e_S,
	1.0,
	lin_c_S,
	"1",
	Dict(fk_in.name=>fk_in),
	r0tvals,
	1.0./r0tvals.^5,
	r0tvals.^6,
	(lpoa = false,
	rta = false,
	lp  = false,
	ptb = true,
	CS = true),
	mb,
	fake_multiK_out,
	median(b_e.times),
	0.0
	)
	)
 

# r(x)= x	
r1tvals=r0tvals

fk_in_1 = (tvals =  r1tvals,
				 Kidx = K_union,
				 r_S = lin_S, 
				 f1_S = sq_c_S, 
				 f2_S = lin_c_S,
				 name="Pg"
				 )

SDP = (active = true, bound = 1.8e7, tvals=r1tvals.^6, w_S = w_S, name="2-norm-square")
Lyapunov = (active = false, bound=0,  tvals=r1tvals.^2, w_S = sq_S, name="Lyapunov-Next-Higher")
exact = (active= false, bound = 0.0, tvals=r1tvals,  w_S = [0.0], name="N/A")
mb = MomentBounds((SDP, Lyapunov, exact))
				 
	
ATr1 = ATRewardFunction(
	(
	lin_S,
	pcmc.eq_exps["lin"].ptb_out.approx,
	sq_c_S,
	"lin",
	Dict(fk_in_1.name=>fk_in_1),
	r1tvals,
	1.0 ./(r1tvals.^5),
	r1tvals.^6,
	(lpoa = true,
	rta = true,
	lp  = false,
	ptb = true,
	CS = true),
	mb,
	fake_multiK_out,
	median(b_union.times),
	0.0
	)
	)


	dict_of_rs = Dict(
		ATr0.name => ATr0,
		ATr1.name => ATr1
	);

	dist_tvals = r0tvals
	w_tvals = dist_tvals.^6

SDP = (active = true, bound = 1.8e7, tvals=dist_tvals.^6, w_S = w_S, name="2-norm-square");
Lyapunov = (active = false, bound=0,  tvals=r1tvals.^2, w_S = sq_S, name="Lyapunov-Next-Higher")
exact = (active= false, bound = 0.0, tvals=dist_tvals, w_S = [0.0], name="N/A");
mb = MomentBounds((SDP, Lyapunov, exact));

	at_in = ATIn(
		(pcmc, 
		dict_of_rs,
		["Pg"], 
		dist_tvals,
		# w_S,
		# w_tvals,
		mb,
		false, 
		[0],
		false,
		true));

	at = init_AT(at_in);

	at_other_methods_dist!(at)
	save(string("test/algorithms/comparisons/data/",
    at.at_in.pcmc.name, "-comparisons.jld2"),
    Dict("at" => at))

	at_other_methods_rs!(at)
	save(string("test/algorithms/comparisons/data/",
    at.at_in.pcmc.name, "-comparisons.jld2"),
    Dict("at" => at))

	at_fixedKruns!(at)
	save(string("test/algorithms/comparisons/data/",
		at.at_in.pcmc.name, "-comparisons.jld2"),
		Dict("at" => at))

		# [1] Infanger, Alex and Glynn, Peter W. "A New Truncation Algorithm for Markov Chain Equilibrium Distributions with Computable Error Bounds".