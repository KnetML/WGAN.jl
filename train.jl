include("models.jl")
include("utils.jl")
using Knet, ArgParse

function main(args)
    s = ArgParseSettings()
    s.description = "WGAN Implementation in Knet"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--embed"; arg_type=Int; default=128; help="word embedding size")
        ("--hidden"; arg_type=Int; default=50; help="LSTM hidden size")
        # ("--mlp"; arg_type=bool; default=false; help="MLP size")
        ("--epochs"; arg_type=Int; default=20; help="number of training epochs")
        ("--report"; arg_type=Int; default=500; help="report period in iters")
        ("--valid"; arg_type=Int; default=10000; help="valid period in iters")
        ("--batchsize"; arg_type=Int; default=100; help="batchsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)

    atype = o[:usegpu] ? KnetArray{Float32} : Array{Float32}
    batchsize = o[:batchsize]

    info("Minibatch Size: $batchsize")
    info("Loading dataset")

    data = loadimgtensors("/home/cem/bedroom")
    batches = minibatch4(data, batchsize)
    bsize = size(data)
    
    info("Dataset size: $bsize")
end

main("--usegpu --batchsize 128")
