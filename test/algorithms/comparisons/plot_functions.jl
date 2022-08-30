struct ATPMethodAtr
  color::RGB{Float64}
  linewidth::Float64
  # approx_plot::Bool
  # tv_bound_plot::Bool
  # pir_bound_plot::Bool
  # label::String
end

struct ATPMethods
  pf::ATPMethodAtr
  rn::ATPMethodAtr
  pi3::ATPMethodAtr
  rta::ATPMethodAtr 
  exit::ATPMethodAtr
  lp::ATPMethodAtr
  lpoa::ATPMethodAtr
  CS::ATPMethodAtr
  CS_min_tv::ATPMethodAtr
  CS_stable::ATPMethodAtr
end

function ATPMethods()
  return ATPMethods(
    ATPMethodAtr(RGB(117/255,233/255,109/255), 4),
    ATPMethodAtr(RGB(255/255,165/255,0/255),2.5),
    ATPMethodAtr(RGB(247/255,205/255,255/255),1),
    ATPMethodAtr(RGB(250/255,99/255,99/255),1),
    ATPMethodAtr(RGB(32/255, 32/255, 171/255),0.5),
    ATPMethodAtr(RGB(104/255,144/255,255/255),2.0),
    ATPMethodAtr(RGB(0,0,0),1),
    ATPMethodAtr(RGB(1,192/255,203/255),1),
    ATPMethodAtr(RGB(30/255,144/255,1),1),
    # ATPMethodAtr(RGB(0, 128/255, 0) ,1),
    # ATPMethodAtr(RGB(175/255, 225/255, 175/255) ,1),
    ATPMethodAtr(RGB(104/255,144/255,255/255),2.0)
    )
end


struct ATPlottingStruct
  nt::Number
  tvals::AbstractArray
  tvals_for_line_plot::AbstractArray 
  tvals_subset_indices_heatmap::AbstractArray
  methods::ATPMethods
end

# Descendant of plotting struct that is more extensible.
ATPlotMetaData = @NamedTuple{
  directory_name::String,
  tvals::Vector{Float64},
  tvals_indices::Vector{Int64},
  nt::Int64
}
ATCompareMethodSeriesData = @NamedTuple{
  color::RGB{Float64},
  label::String,
  data::Vector{Float64},
  timing_data::Vector{Float64}
}
ATCompareMethodPlotData = @NamedTuple{
   nt::Int64,
   tvals::Vector{Float64},
   methods::OrderedDict{String,ATCompareMethodSeriesData}
}


ATPlot = @NamedTuple{
  Meta::ATPlotMetaData,
  Approx::OrderedDict{String,Dict{String,ATCompareMethodPlotData}},
  EqBounds::OrderedDict{String,Dict{String,ATCompareMethodPlotData}},
  TVBounds::OrderedDict{String,Dict{String,ATCompareMethodPlotData}}
}

function compare_approx_plots(atcmpd::ATCompareMethodPlotData)
  data = zeros(atcmpd.nt, length(atcmpd.methods))

  for (i,(k,v)) in enumerate(atcmpd.methods)
    data[:,i] = v.data
  end

  p_err = groupedbar(atcmpd.tvals, 
    log10.(data),
    labels=[v.label for (_,v) in atcmpd.methods],
    linewidth=0.2,
    bar_width=270,
    legend=:bottomleft,
    color=[v.color for (_,v) in atcmpd.methods],
    left_margin=10mm,
    right_margin=10mm,
    ylabel=L"$\log_{10}(\frac{1}{2}||\pi_{\mathrm{apx}}- \pi||_1 )$",
    xlabel= L"$t$ for $A = \{x:e^Tx\leq t\}$",
    title=L"Approximation Errors for $\pi$"
  )
  return p_err
end


function get_xlabel_str(levelname::String)
  if levelname == "level"
    return L"$t$ for $K=\{x: e^Tx = t\}$"
  elseif levelname == "sublevel"
    return L"$t$ for $K=\{x: e^Tx \leq t\}$"
  elseif levelname == "triangle-vertex"
    return L"$t$ for $K=\{(0,0), (t,0), (0,t)\}$"
  else
    ArgumentError("Not known level set name.")
  end
end

function line_plot(arr::AbstractArray, 
                   ylabel_str::Any,
                   pcmcname::String,
                   levelname::String,
                   plot_title::Any,
                   ps::ATPlottingStruct,
                   filename::String,
                   logscale::Symbol;
                   size_arr=nothing)

  if size_arr == nothing
    size_arr = ones(ps.nt)*1
  end

  
  cpal = palette(:jet1, rev=true, ps.nt)
  try 
    dir = string("../../../../../../../../Apps/Overleaf/Truncation/Figures/",
           pcmcname, "/")
    p = plot(title=plot_title,
          legendfontsize=:5, legend=:outerright,
          ylabel=ylabel_str, dpi = 800, 
          left_margin=10mm, right_margin=10mm,
          xlabel=get_xlabel_str(levelname))
    for (i,t) in enumerate(ps.tvals_for_line_plot)
      plot!(ps.tvals_for_line_plot[1:i], arr[i,1:i],
        label=latexstring("A = \\{x:x\\leq",
        ps.tvals_for_line_plot[i],"\\}"), 
        yaxis=logscale,
        color=cpal[i],
        linewidth=size_arr[i])
      scatter!(ps.tvals_for_line_plot[1:i], arr[i,1:i],
        label="", 
        yaxis=logscale,
        markerstrokewidth=0,
        color=cpal[i])
    end
    savefig(p, string(dir, "Lines/", filename,".png"))
    return p
  catch e
    println("Error")
    println(e)
    return
  end
end


function heatmap_plot(arr::AbstractArray, 
                   heatmapval_str::Any,
                   pcmcname::String,
                   levelname::String,
                   plot_title::Any,
                   ps::ATPlottingStruct,
                   filename::String,
                   logscale::Symbol;
                   rev::Bool=false)

  try 
    dir = string("../../../../../../../../Apps/Overleaf/Truncation/Figures/",
           pcmcname, "/")
    p = heatmap(ps.tvals[ps.tvals_subset_indices_heatmap],
            ps.tvals[ps.tvals_subset_indices_heatmap],
            arr[ps.tvals_subset_indices_heatmap, ps.tvals_subset_indices_heatmap],
            legend=false, colorbar=true, linewidth=1,
            dpi = 600,
            xlabel=get_xlabel_str(levelname),
            ylabel=L"$t$ for $A=\{x: e^Tx \leq t\}$",
            title = heatmapval_str,
            c=cgrad(:thermal,rev=rev)
            )
    savefig(p, string(dir, "Heatmaps/", filename,".png"))
    return p
  catch e
    println("Error")
    println(e)
    return
  end
end



function xlevel_apxout_line_plots(apx_out::ATApxOut,                  
                               pcmcname::String,
                               ps::ATPlottingStruct,
                               levelname::String)
                               
  p = line_plot(apx_out.rn.apx_errs, L"TV$(\pi_2^*,\pi)$", 
                pcmcname, levelname, ps, "rn-tv")
  p = line_plot(apx_out.rn.apx_times*1e-9, "Runtime (s)", 
                pcmcname, levelname, ps, "rn-times")

  #pf
  p = line_plot(apx_out.pf.apx_errs, L"TV$(\pi_1^*,\pi)$",
                pcmcname, levelname, ps, "pf-tv")
  p = line_plot(apx_out.pf.apx_times*1e-9, "Runtime (s)", 
                pcmcname, levelname, ps, "pf-times")

  #cond
  p = line_plot(apx_out.cond.apx_errs, L"TV$(\pi_K^*,\pi)$", 
                pcmcname, levelname, ps, "cond-tv")
  # p = line_plot(at_out.apx_out.cond.apx_times*1e-9, "Runtime (s)", ps, at_out, "Level", "cond-times")

  #pi3
  p = line_plot(apx_out.pi3.apx_errs, L"TV$(\pi_3^*,\pi)$", 
                pcmcname, levelname, ps, "pi3-tv")
  p = line_plot(apx_out.pi3.apx_times*1e-9, "Runtime (s)",
                pcmcname, levelname, ps, "pi3-times")

  #exit
  p = line_plot(apx_out.pi3.apx_errs, L"TV$(\pi_{exit}^*,\pi)$",
                pcmcname, levelname, ps, "exit-tv")
  p = line_plot(apx_out.pi3.apx_times*1e-9, "Runtime (s)", 
                pcmcname, levelname, ps, "exit-times")
end

# For later.
  # # Quantities of Interest
  # p = line_plot(aio.condImP22s, L"cond$(I-P_{22})$",
  #               pcmcname, levelname, "condImP22s")
  # p = line_plot(aio.rho_Gs, L"$\rho(G)$",
  #               pcmcname, levelname, ps, "rho_Gs")

function xlevel_atbri_line_plots(atri::ATrBdsOut_i, 
                                 ps::ATPlottingStruct,
                                 pcmcname::String,
                                 levelname::String,
                                 rstr::String,
                                 algstr::String)
  line_plot(atri.rweightedTVs, string(rstr,"-",algstr,"-weighted-TV"), 
             pcmcname, levelname, ps, string(rstr,"-weighted-TV"))
  line_plot(atri.rweightedTVs, string(rstr,"-",algstr,"-weighted-TV-times-(s)"), 
  pcmcname, levelname, ps, string(rstr,"-",algstr,"-weighted-TV-times-(s)"))
  print(atri.ptb_lbs)
  line_plot(atri.ptb_lbs, string(rstr,"-",algstr,"-ptb-lbs"),
             pcmcname, levelname, ps, string(rstr,"-",algstr,"-ptb-lbs"))
  line_plot(atri.ptb_ubs, string(rstr,"-",algstr,"-ptb-ubs"),
  pcmcname, levelname, ps, string(rstr,"-",algstr,"-ptb-ubs"))

  line_plot(atri.ptb_bds_times, string(rstr,"-",algstr,"-ptb-times"),
            pcmcname, levelname, ps, string(rstr,"-",algstr,"-ptb-times"))

  line_plot(atri.cond_Finvs, string(rstr,"-",algstr,"-cond-Finvs"),
            pcmcname, levelname, ps, string(rstr,"-",algstr,"-cond-Finvs"))

  line_plot(atri.Deltas, string(rstr,"-",algstr,"-Deltas"),
            pcmcname, levelname, ps, string(rstr,"-",algstr,"-Deltas"))
  
  line_plot(atri.minrowsum_Finvs, string(rstr,"-",algstr,"-minrowsum-Finv"),
            pcmcname, levelname, ps, string(rstr,"-",algstr,"-minrowsum-Finv"))

  line_plot(atri.minrowsum_Finvs, string(rstr,"-",algstr,"-maxrowsum-Finv"),
            pcmcname, levelname, ps, string(rstr,"-",algstr,"-maxrowsum-Finv"))
          
end

function xlevel_atiorout_plot(atri::ATIOrOut, 
                              ps::ATPlottingStruct,
                              pcmcname::String,
                              levelname::String,
                              rstr::String)
  # line_plot(atri.r_apx_out.)

  xlevel_atbri_line_plots(atri.rn, ps, pcmcname, levelname, rstr, "rn")
  xlevel_atbri_line_plots(atri.pf, ps, pcmcname, levelname, rstr, "pf")
  line_plot(atri.lp_lb, string(rstr,"-LPLB"), pcmcname, 
            levelname, ps, string(rstr,"-LPLB"))
  line_plot(atri.lp_ub, string(rstr,"-LPUB"), pcmcname,
             levelname, ps, string(rstr,"-LPUB"))
  line_plot(atri.lp_times, string(rstr,"-LPtimes"), pcmcname,
             levelname, ps, string(rstr,"-LPtimes"))                 
end

function make_io_line_plots(io_out::ATIOOut, ps::ATPlottingStruct, 
                    pcmcname::String, levelname::String)
  for (key, val) in io_out.routs
    xlevel_atiorout_plot(val, ps, pcmcname,levelname, key)
  end
  xlevel_apxout_line_plots(io_out.apx_out,pcmcname, ps, levelname)
  line_plot(io_out.rho_Gs, "rhoGs", pcmcname, 
            levelname, ps, "rhoGs")
  line_plot(io_out.condImP22s, "condImP22s",
            pcmcname, levelname, ps, "condImP22s")
end

function make_io_line_plots_nobounds(io_out::ATIOOut, ps::ATPlottingStruct, 
                                    pcmcname::String, levelname::String)
    for (key, val) in io_out.routs
      xlevel_atiorout_plot(val, ps, pcmcname,levelname, key)
    end
    xlevel_apxout_line_plots(io_out.apx_out)
    line_plot(io_out.rho_Gs, "rhoGs", pcmcname, 
              levelname, ps, "rhoGs")
    line_plot(io_out.condImP22s, "condImP22s", 
              pcmcname, levelname, "condImP22s")
end

function make_at_line_plots(at::AT, ps::ATPlottingStruct)
  make_io_line_plots(at.IO.sublevel, ps, at.at_in.pcmc.name, "sublevel")
  make_io_line_plots(at.IO.level, ps, at.at_in.pcmc.name, "level")
end

# function heatmap_plot(arr::AbstractArray, 
#                       ylabel_str::Any,
#                       ps::ATPlottingStruct,
#                       at_out::ATOut,
#                       levelname::String,
#                       filename::String)


# end
