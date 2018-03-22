using FileIO, Images, ImageCore, JLD

@everywhere function readimg(dir, width, height, atype)
    img = Images.imresize(FileIO.load(dir), width, height)
    img = atype.(Images.rawview(ImageCore.channelview(img)[1:3, :, :]))
    # if length(img) != 3*width*height TODO: Find an elegant solution for this
    #     println(dir)
    #     rm(dir)
    #     return randn(1, 64, 64, 3)
    # end
    # TODO: Normalize between (-1, 1)
    img = permutedims(img, (2,3,1)) ./ 255
    return reshape(img, 1, width, height, 3)
end

function readimgs(basedir::String, num::Int;
                  extension=".webp", width=64, height=64, atype=Float32)
    imgdirs = readdir(basedir)
    if num == -1
        num = length(imgdirs)
    end
    imgs = @parallel vcat for i = 1:num
        if contains(imgdirs[i], extension)
            imgdir = joinpath(basedir, imgdirs[i])
            readimg(imgdir, width, height, atype)
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

@everywhere function savetensor(tensor, filepath; name="tensor")
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

@everywhere function loadtensor(filepath; name="tensor")
    JLD.jldopen(filepath, "r") do file
        read(file, name)
    end
end

function loadimgtensors(basedir)
    tensordirs = readdir(basedir)
    imgs = @parallel vcat for dir in tensordirs
        loadtensor(joinpath(basedir, dir))
    end
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

# Test stuff
# println("Reading dataset")
# @time myimgs = readimgs("/home/cem/bedroom_train", 4096)
# println(size(myimgs))
#
# println("Save dataset")
# @time saveimgtensors("/home/cem/bedroom", myimgs, 4096)
#
# #println("Load processed dataset")
# @time loadedimgs = loadimgtensors("/home/cem/bedroom")
# println(size(loadedimgs))
