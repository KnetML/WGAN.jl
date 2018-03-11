using FileIO, Images

# TODO: Parallelization
function readimgs(imgdirs::Vector{String}, size...)
    imgs = []
    for imgdir in imgdirs
        append!(imgs, imresize(load(imgdir), size))
    end
    return imgs
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
