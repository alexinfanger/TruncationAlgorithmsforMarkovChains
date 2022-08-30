pcmc = (load("src/models/toggle_switch/data/TS-20-1.jld2"))["pcmc"]
piv_mat = vec_to_twodmatrix(pcmc.mc.vss,pcmc.piv)[1:25,1:25]
heatmap(piv_mat)
dir = string("assets/",pcmc.name,"/")
savefig(string(dir,"TS-20-1-reference-solution.png"))


pcmc = (load("src/models/toggle_switch/data/TS-90-1.jld2"))["pcmc"]
piv_mat = vec_to_twodmatrix(pcmc.mc.vss,pcmc.piv)[1:25,1:25]
heatmap(piv_mat)
dir = string("assets/",pcmc.name,"/")
savefig(string(dir,"TS-90-1-reference-solution.png"))