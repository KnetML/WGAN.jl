using FileIO, Images, ImageCore, JLD, Logging
include(Pkg.dir("Knet","data","imagenet.jl"))

function readimg(dir, width, height, atype)
    img = Images.imresize(FileIO.load(dir), width, height)
    img = atype.(Images.rawview(ImageCore.channelview(img)[1:3, :, :]))
    return img
end

function normalize(x, min, max)
    oldmin = minimum(x)
    oldmax = maximum(x)
    oldrange = oldmax - oldmin
    newrange = max - min

    scale = (x .- oldmin) ./ oldrange
    return scale .* newrange .+ min
end

function readimgs(basedir::String, num::Int;
                  extension=".webp", width=64, height=64, atype=Float32, report=10000)
    imgdirs = readdir(basedir)
    if num == -1
        num = length(imgdirs)
    end
    imgs = Any[]
    for i = 1:num
        i % report == 0 && info("$i/$num Images read.")
        if contains(imgdirs[i], extension)
            imgdir = joinpath(basedir, imgdirs[i])
            img = readimg(imgdir, width, height, atype)
            img = normalize(permutedims(img, (2,3,1)), -1, 1)
            imgs = vcat(imgs, reshape(img, 1, width, height, 3))
        end
    end
    return imgs
end

function samplenoise4(size, n, atype)
    """
    Outputs gaussian noise with size (1, 1, size, n)
    """
    return atype(reshape(randn(size, n), 1, 1, size, n))
end

function savetensor(tensor, filepath; name="tensor")
    JLD.jldopen(filepath, "w") do file
        write(file, name, tensor)
    end
end

function saveimgtensors(basedir, imgs, bsize)
    for k = 1:bsize:size(imgs, 1)
        lo = k
        hi = min(k+bsize-1, size(imgs, 1))
        tensor = imgs[lo:hi,:,:,:]
        println(size(tensor))
        bid = Int(floor(k/bsize))
        filepath = joinpath(basedir, string(bid))*".jld"
        savetensor(tensor, filepath)
    end
end

function loadtensor(filepath; name="tensor")
    JLD.jldopen(filepath, "r") do file
        read(file, name)
    end
end

function loadimgtensors(basedir)
    tensordirs = readdir(basedir)
    imgs = Any[]
    for dir in tensordirs
        imgs = vcat(imgs, loadtensor(joinpath(basedir, dir)))
    end
    return imgs
end

function minibatch4(X, batchsize, atype)
    """
    Size of X is (N, w, h, c)
    Outputs array where each element has size (w, h, c, b)
    """
    data = Any[]
    for i=1:batchsize:size(X, 1)
        limit = min(i+batchsize-1, size(X, 1))
        minibatch = X[i:limit, :, :, :]
        per = permutedims(minibatch, [2, 3, 4, 1]) # Examples are last element
        push!(data, atype(per))
    end
    return data
end

function numparams(paramarr)
    count = 0
    for p in paramarr
        count += length(p)
    end
    return count
end

function generateimgs(generator, params, zsize, atype; n=36, gridsize=(6,6), scale=1.0)
    randz = samplenoise4(zsize, n, atype)
    genimgs = Array(generator(randz, params))
    images = map(i->reshape(genimgs[:,:,:,i], (64, 64, 3)), 1:n)
    return make_image_grid(images; gridsize=gridsize, scale=scale)
end

function saveimgs(imgs; scale=1.0)
    """
    First dimension is number of elements, n has to be square
    """
    n = size(imgs, 1)
    images = map(i->reshape(imgs[i,:,:,:], (64, 64, 3)), 1:n)
    grid = Int(sqrt(n))
    grid = make_image_grid(images; gridsize=(grid, grid), scale=scale)
    save("images.png", colorview(RGB, grid))
end
