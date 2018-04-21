include("utils.jl")
include("layers.jl")
include("params.jl")
using Knet

function dcgeneratorbn(params, moments, x; training=true)
    """
    Deep Convolutional Generator With Batch Notmalization
    """
    x = dcGbn_input(params[1:2], moments[1], x, training)
    x = dcGbn_hidden(params[3:4], moments[2], x, training)
    x = dcGbn_hidden(params[5:6], moments[3], x, training)
    x = dcGbn_hidden(params[7:8], moments[4], x, training)
    return dcGbn_out(params[9], x)
end

function dcgenerator(params, moments, x; training=true)
    """
    Deep Convolutional Generator without Batch Normalization
    Moments and training stay here in order to use other functions
    in a generic way.
    """
    x = dcGinput(params[1], x)
    x = dcGhidden(params[2], x)
    x = dcGhidden(params[3], x)
    x = dcGhidden(params[4], x)
    return dcGout(params[5], x)
end

function dcdiscriminator(params, moments, x, leak; training=true)
    """
    Deep Convolutional Discriminator
    """
    x = dcD(params[1:2], moments[1], x, leak, training)
    x = dcD(params[3:4], moments[2], x, leak, training)
    x = dcD(params[5:6], moments[3], x, leak, training)
    x = dcD(params[7:8], moments[4], x, leak, training)
    return dcDout(params[9], x)
end

function mlpgenerator(params, moments, x; training=true)
    """
    MLP-ReLU Generator
    Moments and training stay here in order to use other functions
    in a generic way. Input is 4D (1,1,zsize,N) squeeze it
    """
    batchsize = size(x, 4)
    zsize = size(x, 3)

    x = mat(x)
    x = mlp(params[1:2], x)
    x = mlp(params[3:4], x)
    x = mlp(params[5:6], x)
    x = mlpoutG(params[7:8], x)
    # Output should be an image
    return reshape(x, 64, 64, 3, batchsize)
end

function mlpdiscriminator(params, moments, x, leak; training=true)
    """
    MLP-ReLU Discriminator
    Moments and training stay here in order to use other functions
    in a generic way. Input is 4D 64,64,3,N. Make input 2D 64x64x3,N
    """
    x = mat(x)
    x = mlp(params[1:2], x)
    x = mlp(params[3:4], x)
    x = mlp(params[5:6], x)
    return mlpoutD(params[7:8], x)
end

# Below G and D are connected

function dcganbnorm(leak, zsize, atype; winit=xavier)
    """
    Regular DCGAN
    """
    gparams, gmoments = dcGinitbn(atype, winit, zsize)
    dparams, dmoments = dcDinitbn(atype, winit)
    return (gparams, gmoments, dcgeneratorbn), (dparams, dmoments, dcdiscriminator)
end

function dcgan(leak, zsize, atype; winit=xavier)
    """
    DCGAN, but generator do not have batch normalization and has constant filter size
    """
    gparams = dcGinit(atype, winit, 256, zsize)
    dparams, dmoments = dcDinitbn(atype, winit)
    return (gparams, nothing, dcgenerator), (dparams, dmoments, dcdiscriminator)
end

function mlpg(leak, zsize, atype; winit=xavier)
    """
    Generator is an MLP, discriminator is DCGAN
    """
    gparams = mlpGinit(atype, winit, 512, zsize)
    dparams, dmoments = dcDinitbn(atype, winit)
    return (gparams, nothing, mlpgenerator), (dparams, dmoments, dcdiscriminator)
end

function mlpgd(leak, zsize, atype; winit=xavier)
    """
    leak stay here in order to use other functions in a generic way.
    """
    gparams = mlpGinit(atype, winit, 512, zsize)
    dparams = mlpDinit(atype, winit, 512)
    return (gparams, nothing, mlpgenerator), (dparams, nothing, mlpdiscriminator)
end
