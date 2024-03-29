include("startup_truncation.jl")
println("Running comparisons: $(Dates.hour(now())):$(Dates.minute(now()))")
include("../test/algorithms/comparisons/at_TS-20-1.jl")
include("../test/algorithms/comparisons/at_TS-90-1.jl")
include("../test/algorithms/comparisons/at_GM1-Unif02p01-mu1.jl")
println("Comparisons complete: $(Dates.hour(now())):$(Dates.minute(now()))")