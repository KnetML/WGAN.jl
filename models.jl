include("utils.jl")
include("layers.jl")
using Knet
# Deep convolutional discriminator
function dcdiscriminator(atype, leak)
    # Input size: (64,64,3)
    return sequential(
        atype,
        conv2d(3, 128, 4, stride=2, padding=1, activation=leakyrelu(leak)), # (32,32,128)
        conv2d(128, 256, 4, stride=2, padding=1, activation=leakyrelu(leak)), # (16,16,256)
        conv2d(256, 512, 4, stride=2, padding=1, activation=leakyrelu(leak)), # (8,8,512)
        conv2d(512, 1024, 4, stride=2, padding=1, activation=leakyrelu(leak)), # (4,4,1024)
        conv2d(1024, 1, 4, activation=sigm), # (1,1,1024)
    )
end

function mlpgan(zsize, leak, atype)
    # Input size zsize
    generator =
        sequential(
            atype,
            dense(zsize, 512, activation=relu),
            dense(512, 512, activation=relu),
            dense(512, 512, activation=relu),
            dense(512, 64*64*3, activation=tanh),
        )
    return generator, dcdiscriminator(atype, leak)
end

function dcgan(zsize, leak, atype)
    # Input size: (1,1,zsize)
    generator =
        sequential(
            atype,
            conv2d(zsize, 1024, 4, transposed=true), # (4,4,1024)
            conv2d(1024, 512, 4, stride=2, padding=1, transposed=true), # (8,8,512)
            conv2d(512, 256, 4, stride=2, padding=1, transposed=true), # (16,16,256)
            conv2d(256, 128, 4, stride=2, padding=1, transposed=true), # (32,32,128)
            conv2d(128, 3, 4, stride=2, padding=1, activation=tanh, transposed=true), # (64,64,3)
    )

    return generator, dcdiscriminator(atype, leak)
end

function dcganbnorm(zsize, leak, atype)
    # Input size: (1,1,zsize)
    generator =
        sequential(
            atype,
            conv2d(zsize, 1024, 4, transposed=true), # (4,4,1024)
            bnorm(1024),
            conv2d(1024, 512, 4, stride=2, padding=1, transposed=true), # (8,8,512)
            bnorm(512),
            conv2d(512, 256, 4, stride=2, padding=1, transposed=true), # (16,16,256)
            bnorm(256),
            conv2d(256, 128, 4, stride=2, padding=1, transposed=true), # (32,32,128)
            bnorm(128),
            conv2d(128, 3, 4, stride=2, padding=1, activation=tanh, transposed=true), # (64,64,3)
    )
    return generator, dcdiscriminator(atype, leak)
end
