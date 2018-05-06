include("models.jl")
include("utils.jl")
include("loss.jl")
include("pyprocess.jl")
using Knet, ArgParse, FileIO, Images

function main(args)
    s = ArgParseSettings()
    s.description = "WGAN Implementation in Knet"

    @add_arg_table s begin
        ("--gpu"; arg_type=Int; default=0; help="GPU ID if -1 don't use GPU")
        ("--dn"; arg_type=Int; default=1; help="Train discriminator n times")
        ("--type"; arg_type=String; default="dcganbn"; help="Type of model one of: [dcganbn (regular DCGAN), mlpg (Generator is MLP),
        mlpgd (Both MLP), dcgan (Generator has no BN and has constant filter size)]")
        ("--data"; arg_type=String; default="/home/cem/lsun"; help="Dataset dir (lmdb)")
        ("--procedure"; arg_type=String; default="gan"; help="Training procedure. gan or wgan")
        ("--zsize"; arg_type=Int; default=100; help="Noise vector dimension")
        ("--epochs"; arg_type=Int; default=20; help="Number of training epochs")
        ("--report"; arg_type=Int; default=500; help="Report loss in n iterations")
        ("--batchsize"; arg_type=Int; default=64; help="Minibatch Size")
        ("--lr"; arg_type=Float64; default=0.0002; help="Learning rate")
        ("--clip"; arg_type=Float64; default=nothing; help="Clip value")
        ("--opt"; arg_type=String; default="adam"; help="Optimizer, one of: [adam, rmsprop]")
        ("--leak"; arg_type=Float64; default=0.2; help="LeakyReLU leak.")
        ("--out"; arg_type=String; default="./models"; help="Output directory for saving model and generating images")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)

    gpuid = o[:gpu]
    atype = gpuid >= 0 ? KnetArray{Float32} : Array{Float32}

    if gpuid >= 0
        gpu(gpuid)
    end

    batchsize = o[:batchsize]
    procedure = o[:procedure]
    zsize = o[:zsize]
    numepoch = o[:epochs]
    modeltype = o[:type]
    leak = o[:leak]
    optimizer = o[:opt]
    lr = o[:lr]
    datadir = o[:data]
    dn = o[:dn]
    report = o[:report]

    myprint("Minibatch Size: $batchsize")
    myprint("Training Procedure: $procedure")
    myprint("Model Type: $modeltype")
    myprint("Noise size: $zsize")
    myprint("Number of epochs: $numepoch")
    myprint("Using $optimizer with learning rate $lr")
    myprint("Dataset directory: $datadir")
    myprint("Training discriminator $dn times")
    myprint("Report steps: $report")
    myprint("Using GPU $gpuid")

    outdir = joinpath(o[:out], modeltype)
    if procedure == "wgan"
        outdir *= "_wgan"
    end
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

    # Form optimizers
    if optimizer == "adam"
        gopt = optimizers(gparams, Adam, lr=lr, beta1=0.5)
        dopt = optimizers(dparams, Adam, lr=lr, beta1=0.5)
    elseif optimizer == "rmsprop"
        gopt = optimizers(gparams, Rmsprop, lr=lr, rho=0.99, eps=1e-8)
        dopt = optimizers(dparams, Rmsprop, lr=lr, rho=0.99, eps=1e-8)
    else:
        throw(ArgumentError("Unknown optimizer"))
    end

    # Save first randomly generated image
    grid = generateimgs(gforw, gparams, gmoments, zsize, atype)
    outfile = joinpath(outdir, "rand.png")
    save(outfile, colorview(RGB, grid))

    modelpath = joinpath(outdir, "model.jld")

    trainD, trainG = traingan(zsize, atype, procedure, o[:clip])

    myprint("Getting data loader...")
    dataset = getdataset(datadir)

    myprint("Started Training...")
    genitertotal = 0

    for epoch in 1:numepoch
        dataiter = getdataiter(dataset, batchsize)
        totaliter = length(dataiter)
        i = 0

        while i < totaliter
            if procedure == "wgan"
                if genitertotal < 25 || genitertotal % 500 == 0
                    Diters = 100
                else
                    Diters = dn
                end

                j = 0;
                while j < Diters && i < totaliter
                    i += 1; j += 1
                    minibatch = atype(getnext(dataiter))
                    dloss = trainD(dparams, gparams, gmoments, dmoments, gforw, dforw, minibatch, dopt, leak)
                end

            elseif procedure == "gan"
                i += 1
                minibatch = atype(getnext(dataiter))
                dloss = trainD(dparams, gparams, gmoments, dmoments, gforw, dforw, minibatch, dopt, leak)
            else
                throw(ArgumentError("Unknown metric"))
            end

            gloss = trainG(gparams, dparams, gmoments, dmoments, gforw, dforw, batchsize, gopt, leak)
            genitertotal += 1
            appendcsv(logdir, gloss, dloss)
            if i % report == 0
                myprint("[$epoch/$numepoch][$i/$totaliter], LossD: $dloss, LossG: $gloss")
            end
       end

       grid = generateimgs(gforw, gparams, gmoments, zsize, atype)
       outfile = joinpath(outdir, "epoch$epoch.png")
       save(outfile, colorview(RGB, grid))
    end

    myprint("Done. Exiting...")
    return 0
end

main(ARGS)
# ===GAN: DCGAN===
# julia train.jl --gpu 0 --type dcganbn
# ===GAN: Generator with no batch norm===
# julia train.jl --gpu 1 --type dcgan
# ===GAN: Generator MLP===
# julia train.jl --gpu 2 --type mlpg
# ===GAN: Both MLP===
# julia train.jl --gpu 3 --type mlpgd


# ===WGAN: DCGAN===
# julia train.jl --gpu 0 --type dcganbn --procedure wgan --clip 0.01 --lr 0.00005 --opt rmsprop --dn 5 --epochs 30
# ===WGAN: Generator with no batch norm===
# julia train.jl --gpu 1 --type dcgan --procedure wgan --clip 0.01 --lr 0.00005 --opt rmsprop --dn 5 --epochs 30
# ===WGAN: Generator MLP===
# julia train.jl --gpu 0 --type mlpg --procedure wgan --clip 0.01 --lr 0.00005 --opt rmsprop --dn 5 --epochs 40
# ===WGAN: Both MLP===
# julia train.jl --gpu 3 --type mlpgd --procedure wgan --clip 0.01 --lr 0.00005 --opt rmsprop --dn 5 --epochs 30
