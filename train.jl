include("models.jl")
include("utils.jl")
using Knet

atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

data = loadimgtensors("/home/cem/bedroom")
batches = minibatch4(data, 128)

println(size(data))
