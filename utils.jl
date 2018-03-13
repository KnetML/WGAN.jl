using FileIO, Images, ImageCore

@everywhere function readimg(dir, width, height, atype)
    img = Images.imresize(FileIO.load(dir), width, height)
    img = atype.(Images.rawview(ImageCore.channelview(img)[1:3, :, :]))
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

@time myimgs = readimgs("/home/cem/bedroom/train", 1024)
println(size(myimgs))
