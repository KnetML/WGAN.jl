include("models.jl")
include("utils.jl")
include("loss.jl")
using Knet, ArgParse, FileIO, Images

function main(args)
    s = ArgParseSettings()
    s.description = "WGAN Implementation in Knet"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--type"; arg_type=String; default="dcganbn"; help="Type of model one of: [dcganbn (regular DCGAN), mlpg (Generator is MLP),
        mlpgb (Both MLP), dcgan (Generator has no BN and has constant filter size)]")
        ("--data"; arg_type=String; default="/home/cem/bedroom"; help="Dataset dir (processed)")
        ("--procedure"; arg_type=String; default="gan"; help="Training procedure. gan or wgan")
        ("--zsize"; arg_type=Int; default=100; help="Noise vector dimension")
        ("--epochs"; arg_type=Int; default=20; help="Number of training epochs")
        ("--report"; arg_type=Int; default=500; help="Report loss in n iterations")
        ("--batchsize"; arg_type=Int; default=64; help="Minibatch Size")
        ("--lr"; arg_type=Any; default=0.0002; help="Learning rate")
        ("--clip"; arg_type=Any; default=nothing; help="Clip value")
        ("--opt"; arg_type=String; default="adam"; help="Optimizer, one of: [adam, rmsprop]")
        ("--leak"; arg_type=Any; default=0.2; help="LeakyReLU leak.")
        ("--out"; arg_type=String; default="/home/cem/WGAN.jl/models"; help="Output directory for saving model and generating images")
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
    optimizer = o[:opt]
    lr = o[:lr]
    datadir = o[:data]

    myprint("Minibatch Size: $batchsize")
    myprint("Training Procedure: $procedure")
    myprint("Model Type: $modeltype")
    myprint("Noise size: $zsize")
    myprint("Number of epochs: $numepoch")
    myprint("Using $optimizer with learning rate $lr")
    myprint("Dataset directory: $datadir")

    o[:usegpu] ? myprint("Using GPU") : myprint("Not using GPU (why)")

    outdir = joinpath(o[:out], modeltype)
    ispath(outdir) || mkdir(outdir)

    logdir = joinpath(outdir, "log.csv")
    isfile(logdir) && rm(logdir)

    # Get model from models.jl
    if modeltype == "dcganbn"
        model = dcganbnorm
    elseif modeltype == "dcgan"
        model = dcgan
    elseif modeltype == "mlpg"
        model = mlpg
    elseif modeltype == "mlpgd"
        model = mlpgd
    else
        throw(ArgumentError("Unknown model type."))
    end

    generator, discriminator = model(leak, zsize, atype)

    gparams, gmoments, gforw = generator
    dparams, dmoments, dforw = discriminator

    gnumparam = numparams(gparams)
    dnumparam = numparams(dparams)
    myprint("Generator # of Parameters: $gnumparam")
    myprint("Discriminator # of Parameters: $dnumparam")

    # Form optimiziers
    if optimizer == "adam"
        gopt = optimizers(gparams, Adam, lr=lr, beta1=0.5)
        dopt = optimizers(dparams, Adam, lr=lr, beta1=0.5)
    elseif opt == "rmsprop"
        gopt = optimizers(gparams, Rmsprop, lr=lr)
        dopt = optimizers(dparams, Rmsprop, lr=lr)
    else:
        throw(ArgumentError("Unknown optimizer"))
    end

    # Save first randomly generated image
    grid = generateimgs(gforw, gparams, gmoments, zsize, atype)
    outfile = joinpath(outdir, "rand.png")
    save(outfile, colorview(RGB, grid))

    modelpath = joinpath(outdir, "model.jld")

    trainD, trainG = traingan(zsize, atype, procedure, o[:clip])

    numchunks = getnumchunks(datadir)
    myprint("Number of chunks: $numchunks")

    myprint("Started Training...")
    for epoch in 1:numepoch
        numelements = 0
        gtotalloss = 0.0
        dtotalloss = 0.0

        for chunk in 1:20:numchunks
            upper = min(numchunks, chunk+20-1) # TODO: Read this from sys args
            myprint("Loading chunks: ($chunk, $upper)")
            data = loadimgtensors(datadir, (chunk, upper))
            numelements += size(data, 1)
            batches = minibatch4(data, batchsize, atype)
            myprint("Fitting chunks")

            for minibatch in batches
                minibatch = atype(minibatch) # Put to GPU one by one so the memory won't explode
                dloss = trainD(dparams, gparams, gmoments, dmoments, gforw, dforw, minibatch, dopt, leak)
                gloss = trainG(gparams, dparams, gmoments, dmoments, gforw, dforw, batchsize, gopt, leak)
                gtotalloss += gloss * batchsize
                dtotalloss += dloss * batchsize
                appendcsv(logdir, gloss, dloss)
           end
       end

       gtotalloss /= numelements
       dtotalloss /= numelements
       elapsed = 0

       myprint("Epoch $epoch took $elapsed: G Loss: $gtotalloss D Loss: $dtotalloss")
       grid = generateimgs(gforw, gparams, gmoments, zsize, atype)
       outfile = joinpath(outdir, "epoch$epoch.png")
       save(outfile, colorview(RGB, grid))
    end

    grid = generateimgs(gforw, gparams, gmoments, zsize, atype)
    outfile = "trained.png"
    save(outfile, colorview(RGB, grid))

    myprint("Done. Exiting...")
    return 0
end
# lr=0.0002 for regular gan
main("--usegpu --type dcganbn --precedure wgan --clip 0.01 --lr 0.00005")
