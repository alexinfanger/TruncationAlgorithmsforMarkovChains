using Distributions
using LinearAlgebra
using QuadGK

struct GM1KParams
    mu::Number
    rv::Sampleable
    rvname::String
    S::Number
end

struct GM1K <: FiniteMarkovChain
    mu::Number
    rv::Sampleable
    rvname::String
    S::Number
    P::AbstractArray
    vss::VectorStateSpace
end

function GM1K(mu::Number, rv::Sampleable, rvname::String, S::Number)
    gm1p = GM1KParams(mu, rv, rvname, S)
    return GM1K(mu, rv, rvname, S, get_P_stable(gm1p), VectorStateSpace([S]))
end

function get_P(gm1p::GM1KParams, tol::Number=1e-16)
  P = zeros(gm1p.S, gm1p.S)
  if gm1p.rvname=="Deterministic"
    println("Deterministic")
    println(gm1p.mu*(gm1p.rv.μ+1))
    pvec = pdf.(Poisson(gm1p.mu*(gm1p.rv.μ+1)),(gm1p.S-1):-1:0)
    println(Poisson(gm1p.mu*(gm1p.rv.μ+1)))
    for i=1:(gm1p.S-1)
      P[i,2:i+1] = pvec[(end-i+1):end]
    end
    P[end, :] = P[end-1,:]
    P[:,1] = 1.0 .-sum(P[:,2:end], dims=2)
  else 
    integrand(t,k)= pdf(gm1p.rv,t)*(gm1p.mu*t)^k*exp(-gm1p.mu*t)/factorial(big(k))
    pvec = zeros(gm1p.S-1)
    for i =(gm1p.S-1):-1:1
      # pvec[end-i+1] = quadgk(t->integrand(t,i-1),minimum(g.r),maximum(g.r))[1]
      pvec[end-i+1] = quadgk(t->integrand(t,i-1),quantile(gm1p.rv,tol),quantile(gm1p.rv,1-tol))[1]
    end
    for i=1:(gm1p.S-1)
      P[i,2:i+1] = pvec[(end-i+1):end]
    end
    P[end, :] = P[end-1,:]
    P[:,1] = 1.0 .-sum(P[:,2:end], dims=2)
  end
  return P
end

function get_P_stable(gm1p::GM1KParams, tol::Float64=1e-16)
  P = zeros(gm1p.S, gm1p.S)
  if gm1p.rvname=="Deterministic"
    println("Deterministic")
    println(gm1p.mu*(gm1p.rv.μ+1))
    pvec = pdf.(Poisson(gm1p.mu*(gm1p.rv.μ+1)),(gm1p.S-1):-1:0)
    println(Poisson(gm1p.mu*(gm1p.rv.μ+1)))
    for i=1:(gm1p.S-1)
      P[i,2:i+1] = pvec[(end-i+1):end]
    end
    P[end, :] = P[end-1,:]
    P[:,1] = 1.0 .-sum(P[:,2:end], dims=2)
  elseif typeof(gm1p.rv) == Uniform{Float64} && gm1p.mu ==1 && gm1p.rv.a==0
    println("Specializing for Unif(0,b) with mu=1...")
    pvec = zeros(gm1p.S-1)
    for i =(gm1p.S-1):-1:1
      pvec[end-i+1] = gamma_inc(i,gm1p.rv.b)[1]/gm1p.rv.b
    end
    for i=1:(gm1p.S-1)
      P[i,2:i+1] = pvec[(end-i+1):end]
    end
    P[end, :] = P[end-1,:]
    P[:,1] = 1.0 .-sum(P[:,2:end], dims=2)
  else 
    integrand(t,k)= pdf(gm1p.rv,t)*(gm1p.mu*t)^k*exp(-gm1p.mu*t)
    pvec = zeros(gm1p.S-1)
    for i =(gm1p.S-1):-1:1
      # pvec[end-i+1] = quadgk(t->integrand(t,i-1),minimum(g.r),maximum(g.r))[1]
      pvec[end-i+1] = (quadgk(t->integrand(t,i-1),quantile(gm1p.rv,tol),quantile(gm1p.rv,1-tol))[1])/factorial(big(i-1))
    end
    for i=1:(gm1p.S-1)
      P[i,2:i+1] = pvec[(end-i+1):end]
    end
    P[end, :] = P[end-1,:]
    P[:,1] = 1.0 .-sum(P[:,2:end], dims=2)
  end
  return P
end

function to_string(gm1k::GM1K)
    return @sprintf("G(%s)/M(mu=%0.2f)/1/%d", gm1k.rvname, gm1k.mu, gm1k.S-2)
end

struct GM1 <: MarkovChain
    mu::Number
    G::Sampleable
    Gname::String
    stat::Sampleable
end

function GM1(mu::Number, G::Sampleable, Gname::String, tol=1e-12)
    if Gname == "Deterministic"
      tol = 1e-15
      G_mean = Statistics.mean(G)
      rho = find_zero(beta->beta-exp(-mu*G_mean*(1-beta)),[0,1-tol])
    else
      ldinv = Statistics.mean(G)
      rho = (ldinv*mu)^-1
      integrand(t, beta)= pdf(G,t)*exp(-mu*t*(1-beta))
      rho = find_zero(beta->beta-quadgk(t->integrand(t,beta),quantile(G,tol),quantile(G,1-tol))[1],[0,1])
    end
    if rho >= 1
        println(rho)
        ArgumentError("Transient G/M/1 Queue")
    else
        return GM1(mu, G, Gname, Geometric(1-rho))
    end
end

function get_stat_dist(gm1::GM1, eps=1e-16)
    q = Integer(quantile(gm1.stat,1-eps))
    stat_dist = zeros(q)
    for i = 1:q
        stat_dist[i] = pdf(gm1.stat, i-1)
    end
    return stat_dist
end


function mean(gm1::GM1)
    return Statistics.mean(gm1.stat)
end


function to_string(gm1::GM1)
    return @sprintf("G(%s)/M(mu=%0.2f)/1", gm1.Gname, gm1.mu)
end


struct PrecomputedGM1 <: PrecomputedFiniteMarkovChain
  mc::GM1K
  mci::GM1
  piv::Vector{Float64}
  err::Float64
  name::String
  eq_exps::Dict{String,PrecomputedEqExp}
end


function to_table(pcmc::PrecomputedGM1)
  fstr = @sprintf("\\begin{frame}[label=current]\\frametitle{%s Info Table} \n", pcmc.name)
  fstr = string(fstr, @sprintf("\\begin{table}\n"))
  fstr = string(fstr, @sprintf("\\begin{tabular}{|l|l|}\n"))
  fstr = string(fstr, @sprintf("\\hline \n"))
  fstr = string(fstr, @sprintf("Model &  \$G/M/1\$ \\\\ \n"))
  fstr = string(fstr, @sprintf("Submodel &  %s \\\\ \n", pcmc.name))

  fstr = string(fstr, @sprintf(" \$G\$ & %s \\\\ \n", pcmc.mc.rvname))
  fstr = string(fstr, @sprintf("\$\\mu\$ & %2.2f \\\\ \n", pcmc.mc.mu))
  fstr = string(fstr, @sprintf("\$\\rho\$ & %2.2f \\\\ \n", 1/(Statistics.mean(pcmc.mc.rv)*(pcmc.mc.mu))))
  fstr = string(fstr, @sprintf("\$p\$ & %2.2f \\\\ \n", pcmc.mci.stat.p))

  fstr = string(fstr,@sprintf("\\hline\n"))
  fstr = string(fstr,@sprintf("\\end{tabular}\n"))
  fstr = string(fstr,@sprintf("\\end{table}\n"))
  fstr = string(fstr,@sprintf("\\end{frame}"))

  open(string("Models/Queues/submodels/slides/", pcmc.name, ".tex"), "w") do io
      write(io,fstr)
  end

  print(fstr)


  return
end