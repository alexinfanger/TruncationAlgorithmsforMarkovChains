a = load("src/models/queues/gm1/Data/GM1-Unif02p01-mu1.jld2")
pcmc = a["pcmc"]
P = pcmc.mc.P
Q = P-I
vss = pcmc.mc.vss
Sidx = collect(1:vss.size)

e_S = vss_func_to_vec(vss, x->1);
lin_S = vss_abspow_vec(vss, 1);
sq_S = vss_abspow_vec(vss, 2);
cube_S = vss_abspow_vec(vss, 3);
quad_S = vss_abspow_vec(vss, 4)

# Confirm the Lyapunov analysis in [1], putting a theoretical apriori upper bound
# on how large of a truncation set we have to look at to determine K. 
c = 300
EV = pcmc.mc.rv.b/2
EV2 = pcmc.mc.rv.b^2/3  + pcmc.mc.rv.b/2
EV3 = pcmc.mc.rv.b^3/4 + pcmc.mc.rv.b^2 +pcmc.mc.rv.b/2
EV4 = (pcmc.mc.rv.b/2) + (7*pcmc.mc.rv.b^2)/3 + (3*pcmc.mc.rv.b^3)/2 + (pcmc.mc.rv.b)^4/5

n_1 = Int64(ceil(c*(1-2*EV+EV2)/(abs(2*c*(1-EV)+1))))
n_2 = Int64(ceil(sqrt(c*EV3/((c-200)*(EV-1)))))
n_union = max(n_1,n_2)
a0 = c*(1-4EV+6*EV2-4EV3+EV4)
a1 = 4c*(1-3EV + 3(EV2)^2-EV3)
a2 = 6c*(1-2*EV+EV2)
a3 = 4c*(1-EV)+1
n_mb = Int64(ceil(1 + max(abs(a2/a3), abs(a1/a3), abs(a0/a3)))) # Cauchy bound

g_1_S = c*lin_S
g_2_S = c*sq_S
g_3_S = c*quad_S

A_e = get_linear_sublevel_set(vss, n_1)
K_e = get_K_rate_matrix_simple_on_subset(Q, e_S, g_1_S, A_e)
b_e = @benchmark get_K_rate_matrix_simple_on_subset($Q, $e_S, $g_1_S, $A_e)
println("Correct Lyapunov function and K pair (r(x)=1):  ", is_Lyapunov_function_rate(Q, setdiff(Sidx, K_e), e_S, g_1_S))


A_union = get_linear_sublevel_set(vss, n_union)
K_union = get_K_union_on_subset(Q, e_S, g_1_S,lin_S, g_2_S, A_union)
b_union = @benchmark get_K_union_on_subset($Q, $e_S, $g_1_S, $lin_S, $g_2_S, $A_union)
println("Correct Lyapunov function and K pair (r(x)=1 and r(x)=x_1+x_2):  ", is_Lyapunov_function_rate(Q, setdiff(Sidx, K_union), lin_S, g_2_S), ", ",is_Lyapunov_function_rate(Q, setdiff(Sidx, K_union), e_S, g_1_S))

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

r0tvals = [500,1000,1500,2000,2500]

fk_in = (tvals =  r0tvals,
				 Kidx = K_e,
				 r_S = e_S, 
				 f1_S = lin_S, 
				 f2_S = lin_S,
				 name="Pg")

SDP = (active = false, bound = 1.8e7, tvals=r0tvals, w_S = [0.0], name="2-norm-square")
Lyapunov = (active=false, bound=0.0,  tvals=r0tvals, w_S = [0.0], name="N/A")
exact = (active= false, bound = 0.0, tvals=r0tvals, w_S = [0.0], name="N/A")
mb = MomentBounds((SDP, Lyapunov, exact))

ATr0 = ATRewardFunction((
	e_S,
	1.0,
	lin_S,
	"1",
	Dict(fk_in.name=>fk_in),
	r0tvals,
	[],
	[],
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
				 f1_S = sq_S, 
				 f2_S = lin_S,
				 name="Pg"
				 )

A_mb = get_linear_sublevel_set(vss, n_mb)
mb_val = LyapunovBound2MomentBoundQ_on_subset(Q, g_3_S, cube_S, A_mb)
b_mb = @benchmark LyapunovBound2MomentBoundQ_on_subset($Q, $g_3_S, $cube_S, $A_mb)
println("confirm moment bound holds: ", mb_val>= pcmc.piv'*cube_S)

SDP = (active = false, bound = 1.8e7, tvals=r1tvals, w_S = [0.0], name="2-norm-square")
Lyapunov = (active = true, bound=mb_val,  tvals=r1tvals.^3, w_S = cube_S, name="Lyapunov-Second-Higher")
exact = (active= false, bound = 0.0, tvals=r1tvals,  w_S = [0.0], name="N/A")
mb = MomentBounds((SDP, Lyapunov, exact))
				 
	
ATr1 = ATRewardFunction(
	(
	lin_S,
	pcmc.eq_exps["lin"].ptb_out.approx,
	sq_S,
	"lin",
	Dict(fk_in_1.name=>fk_in_1),
	r1tvals,
	1.0./(r1tvals.^2), #Peter's new bound
	r1tvals.^3,
	(lpoa = true,
	rta = true,
	lp  = false,
	ptb = true,
	CS=true),
	mb,
	fake_multiK_out,
	median(b_union.times),
	median(b_mb.times)
	)
	)

	dict_of_rs = Dict(
		ATr0.name => ATr0,
		ATr1.name => ATr1
	);

dist_tvals = r0tvals

SDP = (active = false, bound = 1.8e7, tvals=dist_tvals.^6, w_S = [0.0], name="2-norm-square");
Lyapunov = (active = true, bound=mb_val, tvals=r1tvals.^3, w_S = cube_S, name="Lyapunov-Second-Higher")
exact = (active= false, bound = 0.0, tvals=dist_tvals, w_S = [0.0], name="N/A");
mb = MomentBounds((SDP, Lyapunov, exact));

	at_in = ATIn(
		(pcmc, 
		dict_of_rs,
		["Pg"], 
		dist_tvals,
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