using Knet

function conv2d(inchannel, outchannel, kernelsize; stride=1, padding=0,
                activation=relu, transposed=false)
    wsize = transposed ? (kernelsize, kernelsize, outchannel, inchannel) : (kernelsize, kernelsize, inchannel, outchannel)
    bsize = (1, 1, outchannel, 1)
    function forward(x, w, b)
        x = transposed ? deconv4(w, x, padding=padding, stride=stride) : conv4(w, x, padding=padding, stride=stride)
        x = x .+ b
        activation != nothing ? activation.(x) : x
    end
    return wsize, bsize, forward
end

function dense(insize, outsize; activation=nothing)
    wsize = (outsize, insize)
    bsize = (outsize, 1)
    function forward(x, w, b)
        if length(size(x)) != 2 x=mat(x) end
        x = w * x
        x = x .+ b
        activation != nothing ? activation.(x) : x
    end
    return wsize, bsize, forward
end

function sequential(atype, layers...; winit=xavier, binit=zeros)
    """
    `layers...` are the layers implemented in the `layers.jl` file. This layers
    returns 3-element tuples containing weight size, bias size and the forward pass functions.
    This method returns a triplet, first element is array parameters, second element
    is a function that takes inputs `x` and passes x through the network and last
    element is a function to update the parameters given a gradient
    """
    ws = Any[]
    forws = Any[]

    for l in layers
        push!(ws, winit(l[1])) # Init the weight
        push!(ws, binit(l[2])) # Init the bias
        push!(forws, l[3]) # Layer forward
    end

    ws = map(atype, ws)

    function forward(x)
        i = 1
        for f in forws
            x = f(x, ws[i], ws[i+1])
            i += 2
        end
        return x
    end

    function update(grads, optims)
        update!(ws, grads, optims)
    end

    ws, forward, update
end
