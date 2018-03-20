using Knet

function conv2d(inchannel, outchannel, kernelsize; stride=1, padding=0,
                activation=relu, initializer=xavier, transposed=false, bias=true)
    w = initializer(kernelsize, kernelsize, inchannel, outchannel)
    b = zeros(outchannel, 1)
    function forward(x)
        x = transposed ? deconv4(x, w, padding=padding, stride=stride) : conv4(x, w, padding=padding, stride=stride)
        x = bias ? x .+ b : x
        activation != nothing ? activation.(x) : x
    end
    return w, b, forward
end

function dense(insize, outsize; activation=nothing, initializer=xavier, bias=true)
    w = initializer(outsize, insize)
    b = zeros(outsize, 1)
    function forward(x)
        if length(size(x)) x=mat(x) end
        x = w*x
        x = bias ? x .+ b : x
        activation != nothing ? activation.(x) : x
    end
    return w, b, forward
end

function sequential(layers...)
    """
    `layers...` are the layers implemented in the `layers.jl` file. This layers
    returns 3-element tuples containing weight, bias and the forward pass functions.
    This method returns a tuple, first element is array parameters and second element
    is a function that takes inputs `x` and passes x through the network.
    """
    ws = Array{Any}
    forws = Array{Any}
    for l in layers
        push!(ws, l[0])
        push!(ws, l[1])
        push!(forws, l[2])
    end
    function forward(x)
        for f in forws
            x = f(x)
        end
        x
    end
    ws, forward
end
