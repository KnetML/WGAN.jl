using FileIO, Images

function readimgs(imgdirs::Vector{String}, height, width)
    imgs = []
    imgs = @parallel (vcat) for i = 1:length(imgdirs)
        Images.imresize(FileIO.load(imgdirs[i]), height, width)
    end
    return v
end

function getimgdirs(dir::String, extension::String)
    imgs = filter(x->contains(x,extension), readdir(dir))
    imgdirs = map(x->joinpath(dir, x), imgs)
    return imgdirs
end

println("Getting dirs")
imgdirs = getimgdirs("/home/cem/bedroom/train", ".webp")

println("Reading images")
@time readimgs(imgdirs, 64, 64)
