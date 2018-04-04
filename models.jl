include("utils.jl")
include("layers.jl")
include("params.jl")
using Knet

function dcgeneratorbn(params, moments, x; training=true)
    x = dcGbn1(params[1:2], moments[1], x, training)
    x = dcGbn2(params[3:4], moments[2], x, training)
    x = dcGbn3(params[5:6], moments[3], x, training)
    x = dcGbn4(params[7:8], moments[4], x, training)
    return dcGbn5(params[9], x)
end

function dcdiscriminator(params, moments, x, leak; training=true)
    x = dcD1(params[1:2], moments[1], x, leak, training)
    x = dcD2(params[3:4], moments[2], x, leak, training)
    x = dcD3(params[5:6], moments[3], x, leak, training)
    x = dcD4(params[7:8], moments[4], x, leak, training)
    return dcD5(params[9], x)
end

function dcganbnorm(leak, zsize, atype; winit=xavier)
    gparams, gmoments = dcGinitbn(atype, winit, zsize)
    dparams, dmoments = dcDinitbn(atype, winit)
    return (gparams, gmoments, dcgeneratorbn), (dparams, dmoments, dcdiscriminator)
end
