include("models.jl")
include("utils.jl")
using Knet

data = loadimgtensors("/home/cem/bedroom")
batches = minibatch4(data, 128)

for b in batches
    println(size(b))
end
