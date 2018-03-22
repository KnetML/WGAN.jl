include("models.jl")
include("utils.jl")
using Knet, ArgParse, FileIO, Images
include(Pkg.dir("Knet","data","imagenet.jl"))


function main(args)
    s = ArgParseSettings()
    s.description = "WGAN Implementation in Knet"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--bn"; action=:store_true; help="Use batchnorm in generator")
        ("--mlp"; action=:store_true; help="Use 4 layer MLP as generator")
        ("--procedure"; arg_type=String; default="gan"; help="Training procedure. gan or wgan")
        ("--zsize"; arg_type=Int; default=100; help="Noise vector dimension")
        ("--epochs"; arg_type=Int; default=20; help="Number of training epochs")
        ("--report"; arg_type=Int; default=500; help="Report loss in n iterations")
        ("--batchsize"; arg_type=Int; default=128; help="batchsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)

    atype = o[:usegpu] ? KnetArray{Float32} : Array{Float32}

    batchsize = o[:batchsize]
    procedure = o[:procedure]
    ismlp = o[:mlp]
    isbn = o[:bn]
    zsize = o[:zsize]
    numepoch = o[:epochs]

    generatortype = ismlp ? "MLP " : "DCGAN "
    generatortype *= isbn ? "BN" : "No BN"

    info("Minibatch Size: $batchsize")
    info("Training Procedure: $procedure")
    info("Generator Type: $generatortype")
    info("Noise size: $zsize")
    info("Number of epochs: $numepoch")

    o[:usegpu] ? info("Using GPU") : info("Not using GPU (why)")

    info("Loading dataset")

    data = loadimgtensors("/home/cem/bedroom")
    bsize = size(data)

    info("Dataset size: $bsize")

    model = ismlp ? mlpgan : dcgan
    generator, discriminator = model(zsize, atype)
    generator_params, generator_forward, generator_update = generator
    discriminator_params, discriminator_forward, discriminator_update = discriminator

    gnumparam = numparams(generator_params)
    dnumparam = numparams(discriminator_params)
    info("Generator # of Parameters: $gnumparam")
    info("Discriminator # of Parameters: $dnumparam")

    # TODO: Make a function for this
    randz = samplenoise4(zsize, 25, atype)
    genimgs = Array(generator_forward(randz)) .* 255

    images = map(i->reshape(genimgs[:,:,:,i], (64, 64, 3)), 1:25)
    grid = make_image_grid(images; gridsize=(5, 5), scale=1.0)

    outfile = "rand.png"
    save(outfile, colorview(RGB, grid))
    # End of temporary stuff

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
