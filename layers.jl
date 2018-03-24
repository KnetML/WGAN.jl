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

    return wsize, bsize, forward, "conv"
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
    return wsize, bsize, forward, "dense"
end

function bnorm(channels)
    function forward(x, moments, params, training)
        return batchnorm(x, moments, params, training=training)
    end
    return channels, forward, "bn"
end

function sequential(atype, layers...;winit=xavier, binit=zeros)
    """
    `layers...` are the layers implemented in the `layers.jl` file. This layers
    returns 3-element tuples containing weight size, bias size and the forward pass functions.
    This method returns a triplet, first element is array parameters, second element
    is a function that takes inputs `x` and passes x through the network and last
    element is a function to update the parameters given a gradient
    """
    ltypes = Any[]
    params = Any[]
    moments = Any[]
    forws = Any[]

    for l in layers
        ltype = l[end]
        push!(ltypes, ltype)

        if ltype == "bn" # This means batchnorm
            channels = l[1]
            push!(moments, bnmoments()) # Get moments
            push!(params, bnparams(channels)) # Init BN params
            push!(forws, l[2]) # BN forward
        else
            push!(params, winit(l[1])) # Init the weight
            push!(params, binit(l[2])) # Init the bias
            push!(forws, l[3]) # Layer forward
            push!(moments, nothing) # For the sake of indexing done below
        end

    end

    params = map(atype, params)

    function forward(x, params; training=true)
        i = 1 # For parameters
        j = 1 # For moments
        for (l, f) in zip(ltypes, forws)
            if l == "bn"
                x = f(x, moments[j], params[i], training)
                i += 1
            else
                x = f(x, params[i], params[i+1])
                i += 2
            end
            j += 1
        end
        return x
    end
    # Use moments inside the function but return params to use in grad functions
    params, forward
end

function leakyrelu(alpha)
    return x -> (relu(x) - relu(-x) * alpha)
end
