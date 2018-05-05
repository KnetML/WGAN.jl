using Knet

# Deep Convolutional Generator
function dcGbn_input(w, m, x, training)
    x = deconv4(w[1], x, mode=1)
    x = batchnorm(x, m, w[2], training=training)
    return relu.(x)
end

function dcGbn_hidden(w, m, x, training)
    x = deconv4(w[1], x, stride=2, padding=1, mode=1)
    x = batchnorm(x, m, w[2], training=training)
    return relu.(x)
end

function dcGbn_out(w, x)
    x = deconv4(w, x, stride=2, padding=1, mode=1)
    return tanh.(x)
end

# Deep Convolutional Discriminator
# Input does not have batchnorm
function dcDin(w, x, leak)
    x = conv4(w, x, stride=2, padding=1, mode=1)
    return leakyrelu.(x, leak)
end

function dcD(w, m, x, leak, training)
    x = conv4(w[1], x, stride=2, padding=1, mode=1)
    x = batchnorm(x, m, w[2], training=training)
    return leakyrelu.(x, leak)
end

function dcDout(w, x)
    return conv4(w, x, mode=1)
end

# Deep Convolutional Generator with no batchnorm
function dcGinput(w, x)
    x = deconv4(w, x, mode=1)
    return relu.(x)
end

function dcGhidden(w, x)
    x = deconv4(w, x, stride=2, padding=1, mode=1)
    return relu.(x)
end

function dcGout(w, x)
    x = deconv4(w, x, stride=2, padding=1, mode=1)
    return tanh.(x)
end

# Common MLP layer
function mlp(w, x)
    return relu.(w[1] * x .+ w[2])
end

# Common MLP output layer
function mlpout(w, x)
    return w[1] * x .+ w[2]
end

function leakyrelu(x, alpha)
    return max(alpha*x, x)
end
