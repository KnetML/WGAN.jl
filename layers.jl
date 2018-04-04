using Knet

function dcGbn1(w, m, x, training)
    x = deconv4(w[1], x)
    x = batchnorm(x, m, w[2], training=training)
    return relu.(x)
end

function dcGbn2(w, m, x, training)
    x = deconv4(w[1], x, stride=2, padding=1)
    x = batchnorm(x, m, w[2], training=training)
    return relu.(x)
end

function dcGbn3(w, m, x, training)
    x = deconv4(w[1], x, stride=2, padding=1)
    x = batchnorm(x, m, w[2], training=training)
    return relu.(x)
end

function dcGbn4(w, m, x, training)
    x = deconv4(w[1], x, stride=2, padding=1)
    x = batchnorm(x, m, w[2], training=training)
    return relu.(x)
end

function dcGbn5(w, x)
    x = deconv4(w, x, stride=2, padding=1)
    return tanh.(x)
end

function dcD1(w, m, x, leak, training)
    x = conv4(w[1], x, stride=2, padding=1)
    x = batchnorm(x, m, w[2], training=training)
    return leakyrelu.(x, leak)
end

function dcD2(w, m, x, leak, training)
    x = conv4(w[1], x, stride=2, padding=1)
    x = batchnorm(x, m, w[2], training=training)
    return leakyrelu.(x, leak)
end

function dcD3(w, m, x, leak, training)
    x = conv4(w[1], x, stride=2, padding=1)
    x = batchnorm(x, m, w[2], training=training)
    return leakyrelu.(x, leak)
end

function dcD4(w, m, x, leak, training)
    x = conv4(w[1], x, stride=2, padding=1)
    x = batchnorm(x, m, w[2], training=training)
    return leakyrelu.(x, leak)
end

function dcD5(w, x)
    return conv4(w, x)
end

function leakyrelu(x, alpha)
    return max(0,x) + min(0,x) * alpha
end
