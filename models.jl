include("utils.jl")
include("layers.jl")
using Knet
# TODO: Batch normalization
# TODO: Tanh activation at the last layer of generator
function mlpgan(zsize, atype)
    # Input size zsize
    generator =
        sequential(
            atype,
            dense(zsize, 512, activation=relu),
            dense(512, 512, activation=relu),
            dense(512, 512, activation=relu),
            dense(512, 512, activation=relu),
            dense(512, 64*64*3, activation=relu)
        )

    # Input size: (64,64,3)
    # TODO: Activations are LeakyReLU
    discriminator =
        sequential(
            atype,
            conv2d(zsize, 64, 4, stride=2, padding=1), # (32,32,64)
            conv2d(64, 128, 4, stride=2, padding=1), # (16,16,128)
            conv2d(128, 256, 4, stride=2, padding=1), # (8,8,256)
            conv2d(256, 512, 4, stride=2, padding=1), # (4,4,512)
            conv2d(512, 1, 4, activation=sigm), # (1,1,1)
        )
    return generator(), discriminator()
end

function dcgan(zsize, atype)

    # Input size: (1,1,zsize)
    generator =
        sequential(
            atype,
            conv2d(zsize, 512, 4, transposed=true), # (4,4,512)
            conv2d(512, 256, 4, stride=2, padding=1, transposed=true), # (8,8,256)
            conv2d(256, 128, 4, stride=2, padding=1, transposed=true), # (16,16,128)
            conv2d(128, 64, 4, stride=2, padding=1, transposed=true), # (32,32,64)
            conv2d(64, 3, 4, stride=2, padding=1, activation=tanh, transposed=true), # (64,64,3)
    )

    # Input size: (64,64,3)
    # TODO: Activations are LeakyReLU
    discriminator =
        sequential(
            atype,
            conv2d(zsize, 64, 4, stride=2, padding=1), # (32,32,64)
            conv2d(64, 128, 4, stride=2, padding=1), # (16,16,128)
            conv2d(128, 256, 4, stride=2, padding=1), # (8,8,256)
            conv2d(256, 512, 4, stride=2, padding=1), # (4,4,512)
            conv2d(512, 1, 4, activation=sigm), # (1,1,1)
        )
    return generator, discriminator
end
