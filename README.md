### Code for "A new truncation algorithm for Markov chain equilibrium distributions with computable error bounds"

This repository contains the code to reproduce the plots in the paper "A new truncation algorithm for Markov chain equilibrium distributions with computable error bounds" by A. Infanger and P. W. Glynn.

We ran this code on Julia v.1.6.2. The versions of the packages we used can be found
in the project.toml and manifest.toml files, and the correctly versioned packages can be instantiated using Julia's package manager (see code below).

To attempt to reproduce our results, clone the repository and then run the following code:

```
> julia --project="." 

julia> using Pkg
julia> Pkg.instantiate()
julia> include("scripts/reproduce_paper_plots.jl")
```
