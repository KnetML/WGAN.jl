using Knet

function dcGinitbn(atype, winit, zsize)
    w1 = winit(4, 4, 512, zsize)
    m1 = bnmoments()
    b1 = bnparams(512)

    w2 = winit(4, 4, 256, 512)
    m2 = bnmoments()
    b2 = bnparams(256)

    w3 = winit(4, 4, 128, 256)
    m3 = bnmoments()
    b3 = bnparams(128)

    w4 = winit(4, 4, 64, 128)
    m4 = bnmoments()
    b4 = bnparams(64)

    w5 = winit(4, 4, 3, 64)

    params = atype.([w1,b1,w2,b2,w3,b3,w4,b4,w5])
    moments = [m1,m2,m3,m4]
    return params, moments
end

function dcGinit(atype, winit, filtersize, zsize)
    w1 = winit(4, 4, filtersize, zsize)
    w2 = winit(4, 4, filtersize, filtersize)
    w3 = winit(4, 4, filtersize, filtersize)
    w4 = winit(4, 4, filtersize, filtersize)
    w5 = winit(4, 4, 3, filtersize)

    params = atype.([w1,w2,w3,w4,w5])
    return params
end

function mlpGinit(atype, winit, hiddensize, zsize)
    w1 = winit(hiddensize, zsize)
    w2 = winit(hiddensize, hiddensize)
    w3 = winit(hiddensize, hiddensize)
    w4 = winit(hiddensize, hiddensize)
    w5 = winit(64*64*3, hiddensize)

    params = atype.([w1,w2,w3,w4,w5])
    return params
end

function dcDinitbn(atype, winit)
    w1 = winit(4, 4, 3, 64)
    m1 = bnmoments()
    b1 = bnparams(64)

    w2 = winit(4, 4, 64, 128)
    m2 = bnmoments()
    b2 = bnparams(128)

    w3 = winit(4, 4, 128, 256)
    m3 = bnmoments()
    b3 = bnparams(256)

    w4 = winit(4, 4, 256, 512)
    m4 = bnmoments()
    b4 = bnparams(512)

    w5 = winit(4, 4, 512, 1)

    params = atype.([w1,b1,w2,b2,w3,b3,w4,b4,w5])
    moments = [m1,m2,m3,m4]
    return params, moments
end

function mlpDinit(atype, winit, hiddensize)
    w1 = winit(hiddensize, 64*64*3)
    w2 = winit(hiddensize, hiddensize)
    w3 = winit(hiddensize, hiddensize)
    w4 = winit(hiddensize, hiddensize)
    w5 = winit(1, hiddensize)

    params = atype.([w1,w2,w3,w4,w5])
    return params
end
