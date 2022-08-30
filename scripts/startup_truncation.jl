println("Launching startup_truncation.jl ...")

using BenchmarkTools
using Combinatorics
using DataStructures
using Distributions
using FileIO
using JLD2
using LaTeXStrings
using LinearAlgebra
using Plots
using Plots.PlotMeasures
pyplot(dpi=500)
using Printf
using Roots
using StatsPlots
using SpecialFunctions
using SparseArrays
using Arpack
using Statistics

using Convex
using JuMP
using MosekTools
using QuadGK
using Dates


# Models
include("../src/models/load_models.jl")

# Various Functions
include("../src/functions/Markov_process_functions.jl")
include("../src/functions/Doob_transform.jl")
include("../src/functions/Lyapunov_functions.jl")

# Truncation Algorithms
include("../src/algorithms/rta/rta.jl")
include("../src/algorithms/ratio/ratio_full.jl")
include("../src/algorithms/lpoa/lpoa.jl")
include("../src/algorithms/lpoa/strict_interior.jl")

# Tests
include("../test/algorithms/comparisons/at_data_structures.jl")
include("../test/algorithms/comparisons/all_tests.jl")
include("../test/algorithms/comparisons/plot_functions.jl")

println("startup_truncation.jl complete.")