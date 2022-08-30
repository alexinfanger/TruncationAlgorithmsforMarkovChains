"""
    LPOA(
      Q_A::AbstractArray, 
      NrIdx_A::AbstractArray,
      c::Number,
      r::Number,
      w_A::AbstractArray;
      precondition::Bool=true,
      saveall::Bool=false,
      solver::MOI.OptimizerWithAttributes=MOI.OptimizerWithAttributes(Mosek.Optimizer,("QUIET" => true, "MSK_IPAR_OPTIMIZER" => 4)...),
      jump::Bool=true)

LPOA -- Linear Programming Outer Approximation. This function is for computing an 
approximation ``(\\pi(x):x\\in A)``. 


See [1] and [2]. Note [1] says maximizing the total mass is a good heuristic
when trying to find good approximations so we follow this approach here. 

[1] Kuntz, Juan, et al. "Stationary distributions of continuous-time Markov chains: a review of theory and truncation-based approximations." SIAM Review 63.1 (2021): 3-64.
[2] Kuntz, Juan, et al. "Approximations of countably infinite linear programs over bounded measure spaces." SIAM Journal on Optimization 31.1 (2021): 604-625.
"""
function LPOA(
  Q_A::AbstractArray, 
  NrIdx_A::AbstractArray,
  c::Number,
  r::Number,
  w_A::AbstractArray;
  precondition::Bool=true,
  saveall::Bool=false,
  solver::MOI.OptimizerWithAttributes=MOI.OptimizerWithAttributes(Mosek.Optimizer,("QUIET" => true, "MSK_IPAR_OPTIMIZER" => 4)...),
  jump::Bool=true)
				  
  if jump 
    m = direct_model(solver)

    n = size(Q_A,1)
    e = ones(n)
    if precondition
      d = abs.(diag(Q_A))
      Dinv_A = spdiagm(1.0 ./d)
      DinvQ_A = Dinv_A*Q_A
      @variable(m, Dpiv[1:n])
      @constraint(m, piQzero, Dpiv'*DinvQ_A[:, NrIdx_A].==0)
      @constraint(m, moment1, 1-c/r .<=e'*Dinv_A*Dpiv)
      @constraint(m, moment2, e'*Dinv_A*Dpiv.<= 1)
      @constraint(m, moment3, Dpiv'*Dinv_A*w_A <= c)
      @constraint(m, ng, Dpiv.>=0)

      @objective(m, Max, e'*Dinv_A*Dpiv)
      optimize!(m)

      if saveall 
        return m
      else
        return Dinv_A*value.(Dpiv)
      end
      
  else 
    @variable(m, piv[1:n])
    @constraint(m, piQzero, piv'*Q_A[:, NrIdx_A].==0)
    @constraint(m, moment1, 1-c/r .<=e'piv)
    @constraint(m, moment2, e'piv.<= 1)
    @constraint(m, moment3, piv'*w_A <= c)
    @constraint(m, ng, piv.>=0)

    @objective(m, Max, e'*piv)
    optimize!(m)
    if saveall 
      return m
    else
      return value.(piv)
    end
  end

  else 
  # CVX VERSION

    n = size(Q_A,1)
    e = ones(n)
    if precondition
      d = abs.(diag(Q_A))
      Dinv_A = spdiagm(1.0 ./d)
      DinvQ_A = Dinv_A*Q_A
      Dpiv = Variable(n)
      p = maximize(e'*Dinv_A*Dpiv)
      # @variable(m, Dpiv[1:n])
      p.constraints += Dpiv'*DinvQ_A[:, NrIdx_A].==0
      p.constraints += 1-c/r .<=e'*Dinv_A*Dpiv
      p.constraints += e'*Dinv_A*Dpiv.<= 1
      p.constraints += Dpiv'*Dinv_A*w_A <= c
      p.constraints ++ Dpiv.>=0

      # @objective(m, Max, e'*Dinv_A*Dpiv)
      Convex.solve!(p, Mosek.Optimizer, silent_solver=true)

      if saveall 
        return p
      else
        return Dinv_A*Dpiv.value
      end
      
    else 
      piv=Variable(n)
      p = maximize(e'*piv)
      p.constraints += (piv'*Q_A[:, NrIdx_A])==0
      p.constraints += 1-c/r <=e'piv
      p.constraints += e'piv <= 1
      p.constraints += piv'*w_A <= c
      p.constraints += piv >=0

      Convex.solve!(p, Mosek.Optimizer, silent_solver=true)
      if saveall 
        return p
      else
        return piv.value
      end
    end
  end
end

"""
    LPOA_Bound(
      Q_A::AbstractArray, 
      NrIdx_A::AbstractArray,
      c::Number,
      r::Number,
      w_A::AbstractArray,
      r_Aidx::AbstractArray;
      upper::Bool=false,
      saveall::Bool=false,
      print_output=false,
      precondition::Bool=true, 
      solver::MOI.OptimizerWithAttributes=MOI.OptimizerWithAttributes(Mosek.Optimizer,("QUIET" => true, "MSK_IPAR_OPTIMIZER" => 4)...),
      jump::Bool=true)

LPOA -- Linear Programming Outer Approximation. This function is for computing a 
bound on ``\\sum_{x\\in A}\\pi(x)r(x)``. See [1] and [2].

[1] Kuntz, Juan, et al. "Stationary distributions of continuous-time Markov chains: a review of theory and truncation-based approximations." SIAM Review 63.1 (2021): 3-64.
[2] Kuntz, Juan, et al. "Approximations of countably infinite linear programs over bounded measure spaces." SIAM Journal on Optimization 31.1 (2021): 604-625.
"""
function LPOA_Bound(
  Q_A::AbstractArray, 
  NrIdx_A::AbstractArray,
  c::Number,
  r::Number,
  w_A::AbstractArray,
  r_Aidx::AbstractArray;
  upper::Bool=false,
  saveall::Bool=false,
  print_output=false,
  precondition::Bool=true, 
  solver::MOI.OptimizerWithAttributes=MOI.OptimizerWithAttributes(Mosek.Optimizer,("QUIET" => true, "MSK_IPAR_OPTIMIZER" => 4)...),
  jump::Bool=true)

  if jump 
    m = direct_model(solver)
    n = size(Q_A,1)
    e = ones(n)
    if precondition
      d = abs.(diag(Q_A))
      Dinv_A = spdiagm(1.0 ./d)
      DinvQ_A = Dinv_A*Q_A
      @variable(m, Dpiv[1:n])
      @constraint(m, piQzero, Dpiv'*DinvQ_A[:, NrIdx_A].==0)
      @constraint(m, moment1, 1-c/r .<=e'*Dinv_A*Dpiv)
      @constraint(m, moment2, e'*Dinv_A*Dpiv.<= 1)
      @constraint(m, moment3, Dpiv'*Dinv_A*w_A <= c)
      @constraint(m, ng, Dpiv.>=0)
      if upper 
        @objective(m, Max, r_Aidx'*Dinv_A*Dpiv)
      else
        @objective(m, Min, r_Aidx'*Dinv_A*Dpiv)
      end
      optimize!(m)
      if saveall 
        return m
      else
        return Dinv_A*value.(Dpiv)
      end
    else
      @variable(m, piv[1:n])
      @constraint(m, piQzero, piv'*Q_A[:, NrIdx_A].==0)
      @constraint(m, moment1, 1-c/r .<=e'piv)
      @constraint(m, moment2, e'piv.<= 1)
      @constraint(m, moment3, piv'*w_A <= c)
      @constraint(m, ng, piv.>=0)

      if upper 
        @objective(m, Max, r_Aidx'*piv)
      else
        @objective(m, Min, r_Aidx'*piv)
      end
      optimize!(m)
      if saveall 
        return m
      else
        return value.(piv)
      end   
    end       
  else 
    # CVX Version  
    if print_output
      solver=Mosek.solver
    else
      solver = MOI.OptimizerWithAttributes(Mosek.Optimizer,
        #  "MAX_NUM_WARNINGS" => 0,
        "QUIET" => true,
        "MSK_IPAR_OPTIMIZER"=>6)
    end

    n = size(Q_A,1)
    e = ones(n)
    piv = Variable(n)
    if upper 
      p = maximize(r_Aidx'*piv)
    else
      p = minimize(r_Aidx'*piv)
    end
    p.constraints += piv'*Q_A[:, NrIdx_A]==0
    p.constraints += 1-c/r <=e'piv
    p.constraints += e'piv<= 1
    p.constraints += piv'*w_A <= c
    p.constraints += piv>=0

    Convex.solve!(p, Mosek.Optimizer, silent_solver=true)

    if saveall 
      return m
    else
      return piv.value
    end
  end
end


# OLD CODE

# Gurobi benchmark versions.
# function LPOA(Q_A::AbstractArray, 
#                   NrIdx_A::AbstractArray,
#                   c::Number,
#                   r::Number,
#                   w_A::AbstractArray,
#                   Gurobi_ENV;
#                   precondition::Bool=false,
#                   saveall::Bool=false)
				  
# 	# m = Model(GLPK.Optimizer)		
#   m = Model(() -> Gurobi.Optimizer(Gurobi_ENV))
#   set_optimizer_attribute(m, "OutputFlag", 0)

#   n = size(Q_A,1)
# 	e = ones(n)
#   if precondition
#     d = abs.(diag(Q_A))
#     Dinv_A = diagm(1.0 ./d)
#     DinvQ_A = Dinv_A*Q_A
#     @variable(m, Dpiv[1:n])
#     @constraint(m, piQzero, Dpiv'*DinvQ_A[:, NrIdx_A].==0)
#     @constraint(m, moment1, 1-c/r .<=e'*Dinv_A*Dpiv)
#     @constraint(m, moment2, e'*Dinv_A*Dpiv.<= 1)
#     @constraint(m, moment3, Dpiv'*Dinv_A*w_A <= c)
#     @constraint(m, ng, Dpiv.>=0)

#     @objective(m, Max, e'*Dinv_A*Dpiv)
#     optimize!(m)

#     if saveall 
#       return m
#     else
#       return Dinv_A*value.(Dpiv)
#     end
    
#   else 
#     @variable(m, piv[1:n])
#     @constraint(m, piQzero, piv'*Q_A[:, NrIdx_A].==0)
#     @constraint(m, moment1, 1-c/r .<=e'piv)
#     @constraint(m, moment2, e'piv.<= 1)
#     @constraint(m, moment3, piv'*w_A <= c)
#     @constraint(m, ng, piv.>=0)

#     @objective(m, Max, e'*piv)
#     optimize!(m)
#     if saveall 
#       return m
#     else
#       return value.(piv)
#     end
#   end
# end

# function LPOA_Bound(Q_A::AbstractArray, 
#                         NrIdx_A::AbstractArray,
#                         c::Number,
#                         r::Number,
#                         w_A::AbstractArray,
#                         r_Aidx::AbstractArray,
#                         Gurobi_Env;
#                         upper::Bool=false,
#                         saveall::Bool=false)

#   m = Model(() -> Gurobi.Optimizer(Gurobi_Env))
#   set_optimizer_attribute(m, "OutputFlag", 0)

#   n = size(Q_A,1)
#   e = ones(n)
#   @variable(m, piv[1:n])
#   @constraint(m, piQzero, piv'*Q_A[:, NrIdx_A].==0)
#   @constraint(m, moment1, 1-c/r .<=e'piv)
#   @constraint(m, moment2, e'piv.<= 1)
#   @constraint(m, moment3, piv'*w_A <= c)
#   @constraint(m, ng, piv.>=0)

#   if upper 
#     @objective(m, Max, r_Aidx'*piv)
#   else
#     @objective(m, Min, r_Aidx'*piv)
#   end
#   optimize!(m)

#   if saveall 
#     return m
#   else
#     return value.(piv)
#   end          
# end
