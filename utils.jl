using FileIO, Images, ImageCore, JLD
include(Pkg.dir("Knet","data","imagenet.jl"))

@everywhere function myprint(msg::String)
    println(msg); flush(STDOUT)
end

function appendcsv(dir::String, elements...)
    line = join(elements, ",")
    open(dir, "a") do f
        write(f, "$line\n")
    end
end

function samplenoise4(size, n, atype)
    """
    Outputs gaussian noise with size (1, 1, size, n)
    """
    return atype(reshape(randn(size, n), 1, 1, size, n))
end

function numparams(paramarr)
    count = 0
    for p in paramarr
        count += length(p)
    end
    return count
end

function generateimgs(generator, params, moments, zsize, atype; n=36, gridsize=(6,6), scale=2.0)
    randz = samplenoise4(zsize, n, atype)
    genimgs = Array(generator(params, moments, randz, training=false)) .+ 0.5 # de-normalize
    images = map(i->reshape(genimgs[:,:,:,i], (64, 64, 3)), 1:n)
    return make_image_grid(images; gridsize=gridsize, scale=scale)
end

function saveimgs(imgs; scale=1.0)
    """
    First dimension is number of elements, n has to be square
    """
    n = size(imgs, 4)
    images = map(i->reshape(imgs[:,:,:,i], (64, 64, 3)), 1:n)
    grid = Int(sqrt(n))
    grid = make_image_grid(images; gridsize=(grid, grid), scale=scale)
    save("images.png", colorview(RGB, grid))
end
