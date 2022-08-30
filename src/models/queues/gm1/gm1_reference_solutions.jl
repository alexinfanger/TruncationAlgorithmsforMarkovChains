function make_precomputed_gm1_Pg(
  rv::Sampleable, 
  rvname::String, 
  mu::Float64,
  Aidx::Vector{Int64},
  GM1name::String,
  K::Int64,
  lin_scalar::Float64,
  e_scalar::Float64
)
  gi = GM1(mu, rv, rvname)
  # gi_piv_S = get_stat_dist(gi)
  gf = GM1K(mu, rv, rvname, K)
  gf_piv_S = get_stat_dist(gf)
 
  P = gf.P
  vss = gf.vss
  Sidx = collect(1:vss.size)
  SmAsize = size(setdiff(Sidx, Aidx),1)
  if SmAsize==0
    throw(ArgumentError("Sidx-Aidx is empty."))
  else
    println("S\\A Size:   ", size(setdiff(Sidx, Aidx),1))
  end

  # Precomputed solution based on our algorithm
  e_S = vss_func_to_vec(vss, x->1);
  lin_S = vss_abspow_vec(vss, 1);
  sq_S = vss_abspow_vec(vss, 2);

  K_e = findall((P-I)*e_scalar*lin_S.>-e_S)
  println(K_e)
  K_1 = findall((P-I)*lin_scalar*sq_S.>-lin_S)
  println(K_1)
  K_u = union(K_e, K_1)

  rn_out = approx(K_e, Aidx, P, row_normalize)

  println("tv(rn, gf):  ", tv(rn_out.piv_S, gf_piv_S))
  # println("tv(gf, gi):  ", tvzp(gf_piv_S, gi_piv_S))
  # println("tv(rn, gi):  ", tvzp(rn_out.piv_S, gi_piv_S))

  h1_S = optimal_h(P, sq_S, setdiff(Sidx, Aidx))
  h2_S = optimal_h(P, lin_S, setdiff(Sidx, Aidx))

  out_e = CS_tv_bounds(K_e, Aidx, P, e_S, h2_S, h2_S, row_normalize)
  println("e-weighted TV:  ", out_e.tvb)

  out_lin = CS_tv_bounds(K_u, Aidx, P, lin_S, h1_S, h2_S, row_normalize)
  println("lin-weighted TV:  ", out_lin.tvb)

  eq_exp_e = PrecomputedEqExp((out_e, "one"))
  eq_exp_lin =  PrecomputedEqExp((out_lin, "lin"))

  gm1 = PrecomputedGM1(gf, gi, rn_out.piv_S, out_e.tvb, GM1name,
                      Dict(eq_exp_e.name=>eq_exp_e, 
                          eq_exp_lin.name=>eq_exp_lin)
                      )
  save(string("src/models/queues/gm1/data/",GM1name,".jld2"),
                Dict("pcmc" => gm1))
  return K_e, K_u
end

K_e, K_u = make_precomputed_gm1_Pg(Uniform(0.0,2.01), "Uniform(0,3)", 1.0, 
    collect(1:10000), "GM1-Unif02p01-mu1", 20000, 300.0, 300.0)


    