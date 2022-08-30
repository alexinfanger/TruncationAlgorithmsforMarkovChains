# Does not allow changing of denominator currently.
ToggleSwitchParams = @NamedTuple begin
    br::Float64
    dr::Float64
    vss::VectorStateSpace
end

struct ReflectedToggleSwitch <: FiniteMarkovJumpProcess
    br::Float64
    dr::Float64
    vss::VectorStateSpace
    Q::SparseMatrixCSC{Float64, Int64}
end

struct PrecomputedToggleSwitch <: PrecomputedFiniteMarkovJumpProcess
    mc::ReflectedToggleSwitch
    piv::Vector{Float64}
    err::Float64
    name::String
    eq_exps::Dict{String,PrecomputedEqExp}
end

function to_string(rts::ReflectedToggleSwitch)
    return @sprintf("ReflectedToggleSwitch(lambda=%0.2f,mu=%0.2f,%d)", rts.br,
                        rts.dr, rts.vss.array_of_sizes[1])
end


function get_Q_TS(tsp::ToggleSwitchParams)
    rows = []
    cols = []
    vals = Array{Float64, 1}(UndefInitializer(), 0)
    for i=1:tsp.vss.array_of_sizes[1]
        for j=1:tsp.vss.array_of_sizes[2]
            sm = 0.0
            rowind = vec_to_ind(tsp.vss, [i,j])

            if i<tsp.vss.array_of_sizes[1]

                rate = tsp.br/(1+j-1)
                append!(rows, rowind)
                append!(cols, vec_to_ind(tsp.vss,[i+1,j]))
                append!(vals, rate)
                sm += rate
            end

            if i>1
                rate = tsp.dr*(i-1)
                append!(rows, rowind)
                append!(cols, vec_to_ind(tsp.vss,[i-1,j]))
                append!(vals, rate)
                sm += rate
            end

            if j<tsp.vss.array_of_sizes[2]
                rate = tsp.br/(1+i-1)
                append!(rows, rowind)
                append!(cols, vec_to_ind(tsp.vss,[i,j+1]))
                append!(vals, rate)
                sm += rate
            end

            if j>1
                rate = tsp.dr*(j-1)
                append!(rows, rowind)
                append!(cols, vec_to_ind(tsp.vss,[i,j-1]))
                append!(vals, rate)
                sm += rate
            end

            append!(rows, rowind)
            append!(cols, rowind)
            append!(vals, -sm)
        end
    end
    return sparse(rows,cols,vals)
end

function get_Q_TS_Kuntz()
    vss = VectorStateSpace([240,240])
    tsp = ToggleSwitchParams((20.0, 1.0, vss))
    return get_Q_TS(tsp)
end

