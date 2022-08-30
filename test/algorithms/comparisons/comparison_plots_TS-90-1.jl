a = load("test/algorithms/comparisons/data/TS-90-1-comparisons.jld2")
at = a["at"]

println("Starting Making Plots:   ", at.at_in.pcmc.name)

# Paper
# dir = string("../../../../../../../../Apps/Overleaf/EqBoundsPaper/Figures/",
# at.at_in.pcmc.name, "/")

# Current dir
dir = string("assets/",at.at_in.pcmc.name,"/")
mkpath(string(dir, "Comparisons"))  

p = vss_idx_to_heatmap(at.at_in.pcmc.mc.vss, at.at_in.rs["lin"].fk_ins["Pg"].Kidx, maxsize=25)
savefig(p, string(dir, "Kidx.png"))

ps = ATPlottingStruct(
size(at.at_in.tvals,1),
at.at_in.tvals,
at.at_in.tvals,
[1],
ATPMethods()
)

#  Approximations
    nt = size(at.at_in.tvals,1)
    data = zeros(nt, 3)
    e_out = at.IO.fixedK["Pg"].rout["1"]

    data[:,1] = at.IO.fixedK["Pg"].apx_out.rn.apx_errs
    data[:,2] = at.rta.dist.approx_tvs
    data[:,3] = at.lpoa.dist.approx_tvs.mb.Lyapunov

    p = groupedbar(at.at_in.tvals, 
      log10.(data),
      labels=[L"$\pi_2^*$" "RTA" "LPOA"],
      linewidth=0.2,
      bar_width=2.7,
      legend=:bottomleft,
      color=[ps.methods.CS_stable.color ps.methods.rta.color ps.methods.lpoa.color],
      left_margin=10mm,
      right_margin=10mm,
      ylabel=L"$\log_{10}(\frac{1}{2}||\pi_{\mathrm{apx}}- \pi||_1 )$",
      xlabel= L"$t$ for $A = \{x:e^Tx\leq t\}$",
      title=L"Approximation Errors for $\pi$"
    )

    savefig(p, string(dir, "Comparisons/", "Approximation-Bar",".png"))

    timedata = zeros(nt, 3)
    timedata[:,1] = max.(e_out.rn.CS_rweightedTVstimes,at.IO.fixedK["Pg"].apx_out.rn.apx_times)*1e-6 #including runtime for bounds for fairness, I take the max because sometimes (due to noise, I believe), the runtime for the approximation dominates the runtime
    timedata[:,2] = at.rta.dist.approx_times*1e-6
    timedata[:,3] = at.lpoa.dist.times*1e-6
    

    p = groupedbar(at.at_in.tvals, 
      log10.(timedata),
      labels=[L"$\pi_2^*$" "RTA" "LPOA"],
      linewidth=0.2,
      bar_width=2.7,
      color=[ps.methods.CS_stable.color ps.methods.rta.color ps.methods.lpoa.color],
      alpha = [1 1 1],
      left_margin=10mm,
      right_margin=10mm,
      ylabel=L"$\log_{10}$(Runtime) ($\log_{10}$(ms))",
      xlabel= L"$t$ for $A = \{x:e^Tx\leq t\}$",
      title=L"Runtimes for Approximations for $\pi$",
      legend=:topleft
    )
    savefig(p, string(dir, "Comparisons/", "ApproximationTimes-Bar",".png"))



# TV Bounds
  e_out = at.IO.fixedK["Pg"].rout["1"]

  rta_min_bound = min.(at.rta.dist.tv_bound_uppers.mb.Lyapunov,
                      at.rta.dist.tv_bound_lowers.mb.Lyapunov)

  data = zeros(nt, 2)
  # Because our approximation is in fact a distribution, we can take min with 1.
  data[:,1] = min.(e_out.rn.CS_rweightedTVs,1)
  data[:,2] = rta_min_bound


  p = groupedbar(
    at.at_in.tvals, 
    log10.(data),
    labels=[L"Minorization TV Bound ($\pi_2^*$)"  "RTA"],
    linewidth=0.2,
    bar_width=2.7,
    legend=:bottomleft,
    color=[ps.methods.CS_stable.color ps.methods.rta.color],
    left_margin=10mm,
    right_margin=10mm,
    xlabel = L"$t$ for $A = \{x:e^Tx\leq t\}$", 
    ylabel= L"$\log_{10}$(($e$-weighted) TV Bound)",
    title=L"($e$-weighted) TV Bounds"
  )

  savefig(p, string(dir, "Comparisons/", "TVBounds-bar",".png"))

  data[:,1] = (e_out.rn.CS_rweightedTVstimes)*1e-6
  data[:,2] = at.rta.dist.approx_times*1e-6


  p = groupedbar(
    at.at_in.tvals, 
    log10.(data),
    labels=[L"Minorization TV Bound ($\pi_2^*$)"  "RTA"],
    linewidth=0.2,
    bar_width=2.7,
    color=[ps.methods.CS_stable.color ps.methods.rta.color],
    left_margin=10mm,
    right_margin=10mm,
    xlabel = L"$t$ for $A = \{x:e^Tx\leq t\}$", 
    title= L"Runtimes for ($e$-weighted) TV Bounds",
    ylabel=L"$\log_{10}$(Runtime) ($\log_{10}$(ms))",
    legend=:topleft
  )

  savefig(p, string(dir, "Comparisons/", "TVBounds-times-bar",".png"))

# Equilibrium Expectation Bounds 
  nrm = at.at_in.rs["lin"].pir_trueval
  fk_lin_out = at.IO.fixedK["Pg"].rout["lin"]
  rta_lin_out = at.rta.rs["lin"]
  lpoa_lin_out = at.lpoa.rs["lin"]

  data = zeros(nt, 3)*NaN
  data[:,1] = log10.((fk_lin_out.CS_heuristic_ub-fk_lin_out.CS_heuristic_lb)/nrm)
  data[:,2] = log10.((rta_lin_out.ubs.mb.Lyapunov-rta_lin_out.lbs.mb.Lyapunov)/nrm)
  data[:,3] = log10.((lpoa_lin_out.ubs.mb.Lyapunov-lpoa_lin_out.lbs.mb.Lyapunov)/nrm)

  p = groupedbar(
    at.at_in.tvals, 
    data,
    labels=["Minorization Eq. Exp. Bound" "RTA" "LPOA"],
    linewidth=0.2,
    bar_width=2.7,
    color=[ps.methods.CS_stable.color ps.methods.rta.color ps.methods.lpoa.color],
    left_margin=10mm,
    right_margin=10mm,
    xlabel = L"$t$ for $A = \{x:e^Tx\leq t\}$", 
    title= L"Bounds on $\pi r$",
    ylabel=L"$\log_{10}$(Rel. Err. Gap)",
    legend=:bottomleft
  )

  savefig(p, string(dir, "Comparisons/", "pir-bar",".png"))  

  data = zeros(nt, 3)
  data[:, 1] = fk_lin_out.CS_heuristic_times*1e-6
  data[:, 2] = at.rta.dist.approx_times*1e-6
  data[:, 3] = lpoa_lin_out.times*1e-6

  p = groupedbar(at.at_in.tvals, 
    log10.(data),
    labels=["Minorization Eq. Exp. Bound" "RTA" "LPOA"],
    linewidth=0.2,
    bar_width=2.7,
    color=[ps.methods.CS_stable.color ps.methods.rta.color ps.methods.lpoa.color],
    left_margin=10mm,
    right_margin=10mm,
    ylabel=L"$\log_{10}$(runtime) (log(ms))",
    xlabel= L"$t$ for $A = \{x:e^Tx\leq t\}$",
    title=L"Runtimes for Bounds on $\pi r$",
    legend=:topleft
  )

  savefig(p, string(dir, "Comparisons/", "pir-times-bar",".png"))

  println("Printing runtimes of precomputation Lyapunov functions and moment bounds relative to runtimes of algorithms (smallest-truncation-size, largest-truncation-size):")
  println("Lyapunov-time/minimum-row-normalized-approx: ")
  println("   Smallest:   ", (at.at_in.rs["1"].Lyapunov_time + at.at_in.rs["lin"].Lyapunov_time)/(at.IO.fixedK["Pg"].apx_out.rn.apx_times[ps_tval_indices[1]]))
  println("   Largest:   ", (at.at_in.rs["1"].Lyapunov_time + at.at_in.rs["lin"].Lyapunov_time)/(at.IO.fixedK["Pg"].apx_out.rn.apx_times[ps_tval_indices[end]]))

  println("Lyapunov-time/minimum-CS-tv-bound: ")
  println("   Smallest:   ", (at.at_in.rs["1"].Lyapunov_time + at.at_in.rs["lin"].Lyapunov_time)/(at.IO.fixedK["Pg"].rout["1"].rn.CS_rweightedTVstimes[ps_tval_indices[1]]))
  println("   Largest:   ", (at.at_in.rs["1"].Lyapunov_time + at.at_in.rs["lin"].Lyapunov_time)/(at.IO.fixedK["Pg"].rout["1"].rn.CS_rweightedTVstimes[ps_tval_indices[end]]))

  println("Moment-bound-time/row-normalized-approx: ")
  println("   Smallest:   ", at.at_in.rs["lin"].mb_time/(at.IO.fixedK["Pg"].apx_out.rn.apx_times[ps_tval_indices[1]]))
  println("   Largest:   ", at.at_in.rs["lin"].mb_time/(at.IO.fixedK["Pg"].apx_out.rn.apx_times[ps_tval_indices[end]]))

  println("Moment-bound-time/minimum-CS-tv-bound: ")
  println("   Smallest:   ", (at.at_in.rs["lin"].mb_time/(at.IO.fixedK["Pg"].rout["1"].rn.CS_rweightedTVstimes[ps_tval_indices[1]])))
  println("   Largest:   ", (at.at_in.rs["lin"].mb_time/(at.IO.fixedK["Pg"].rout["1"].rn.CS_rweightedTVstimes[ps_tval_indices[end]])))
  
println("Ending Making Plots")



