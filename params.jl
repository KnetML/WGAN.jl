using Knet

function mybn(channels, mean=1.0, std=0.02)
    bnpars = bnparams(channels)
    bnpars[1:channels] = randn(channels) .* std .+ mean
    return bnpars
end

function mlpinit(out, in)
    w = xavier(out, in)
    b = zeros(out, 1)
    return w, b
end

function dcGinitbn(atype, winit, zsize)
    w1 = winit(4, 4, 512, zsize)
    m1 = bnmoments()
    b1 = mybn(512)

    w2 = winit(4, 4, 256, 512)
    m2 = bnmoments()
    b2 = mybn(256)

    w3 = winit(4, 4, 128, 256)
    m3 = bnmoments()
    b3 = mybn(128)

    w4 = winit(4, 4, 64, 128)
    m4 = bnmoments()
    b4 = mybn(64)

    w5 = winit(4, 4, 3, 64)

    params = [w1,b1,w2,b2,w3,b3,w4,b4,w5]
    moments = [m1,m2,m3,m4]
    return atype.(params), moments
end

function dcGinit(atype, winit, filtersize, zsize)
    w1 = winit(4, 4, 512, zsize)
    w2 = winit(4, 4, 256, 512)
    w3 = winit(4, 4, 128, 256)
    w4 = winit(4, 4, 64, 128)
    w5 = winit(4, 4, 3, 64)

    params = [w1,w2,w3,w4,w5]
    return atype.(params)
end

function mlpGinit(atype, hiddensize, zsize)
    w1, b1 = mlpinit(hiddensize, zsize)
    w2, b2 = mlpinit(hiddensize, hiddensize)
    w3, b3 = mlpinit(hiddensize, hiddensize)
    w4, b4 = mlpinit(64*64*3, hiddensize)

    params = [w1,b1,w2,b2,w3,b3,w4,b4]
    return atype.(params)
end

function dcDinitbn(atype, winit)
    w1 = winit(4, 4, 3, 64) # No batchnorm at input layer

    w2 = winit(4, 4, 64, 128)
    m2 = bnmoments()
    b2 = mybn(128)

    w3 = winit(4, 4, 128, 256)
    m3 = bnmoments()
    b3 = mybn(256)

    w4 = winit(4, 4, 256, 512)
    m4 = bnmoments()
    b4 = mybn(512)

    w5 = winit(4, 4, 512, 1)

    params = [w1,w2,b2,w3,b3,w4,b4,w5]
    moments = [m2,m3,m4]
    return atype.(params), moments
end

function mlpDinit(atype, winit, hiddensize)
    w1 = winit(hiddensize, 64*64*3)
    b1 = zeros(hiddensize, 1)
    w2 = winit(hiddensize, hiddensize)
    b2 = zeros(hiddensize, 1)
    w3 = winit(hiddensize, hiddensize)
    b3 = zeros(hiddensize, 1)
    w4 = winit(1, hiddensize)
    b4 = zeros(1, 1)

    params = [w1,b1,w2,b2,w3,b3,w4,b4]
    return atype.(params)
end
