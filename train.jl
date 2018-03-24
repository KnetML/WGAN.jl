include("models.jl")
include("utils.jl")
using Knet, ArgParse, FileIO, Images
include(Pkg.dir("Knet","data","imagenet.jl"))


function main(args)
    s = ArgParseSettings()
    s.description = "WGAN Implementation in Knet"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--type"; arg_type=String; default="dcganbn"; help="Type of model one of: [dcganbn, mlpganbn, dcgan, mlpgan]")
        ("--procedure"; arg_type=String; default="gan"; help="Training procedure. gan or wgan")
        ("--zsize"; arg_type=Int; default=100; help="Noise vector dimension")
        ("--epochs"; arg_type=Int; default=20; help="Number of training epochs")
        ("--report"; arg_type=Int; default=500; help="Report loss in n iterations")
        ("--batchsize"; arg_type=Int; default=128; help="Minibatch Size")
        ("--lr"; arg_type=Any; default=0.0002; help="Learning rate")
        ("--leak"; arg_type=Any; default=0.2; help="LeakyReLU leak.")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)

    atype = o[:usegpu] ? KnetArray{Float32} : Array{Float32}

    batchsize = o[:batchsize]
    procedure = o[:procedure]
    zsize = o[:zsize]
    numepoch = o[:epochs]
    modeltype = o[:type]
    leak = o[:leak]

    info("Minibatch Size: $batchsize")
    info("Training Procedure: $procedure")
    info("Model Type: $modeltype")
    info("Noise size: $zsize")
    info("Number of epochs: $numepoch")

    o[:usegpu] ? info("Using GPU") : info("Not using GPU (why)")

    info("Loading dataset")

    data = loadimgtensors("/home/cem/bedroom")
    bsize = size(data)

    info("Dataset size: $bsize")

    # Get model from models.jl
    if modeltype == "dcganbn"
        model = dcganbnorm
    elseif modeltype == "dcgan"
        model = dcgan
    elseif modeltype == "mlpganbn"
        model = mlpganbnorm
    elseif modeltype == "mlpgan"
        model = mlpgan
    else
        throw(ArgumentError("Unknown model type."))
    end

    generator, discriminator = model(zsize, leak, atype)

    generator_params, generator_forward, generator_update = generator
    discriminator_params, discriminator_forward, discriminator_update = discriminator

    gnumparam = numparams(generator_params)
    dnumparam = numparams(discriminator_params)
    info("Generator # of Parameters: $gnumparam")
    info("Discriminator # of Parameters: $dnumparam")

    grid = generateimgs(generator_forward, zsize, atype)
    outfile = "rand.png"
    save(outfile, colorview(RGB, grid))

    batches = minibatch4(data, batchsize, atype)

    info("Started Training...")
    for epoch in 1:numepoch
        for minibatch in batches
            z = samplenoise4(zsize, batchsize, atype)
            gen = generator_forward(z)
            dis = discriminator_forward(batches[1])
            # TODO: Loss metrics and parameter update
       end
    end

    info("Done. Exiting...")
    return 0
end

main("--usegpu")
