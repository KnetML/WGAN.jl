using Knet

function binaryxentropy(real, logits; average=true)
    loss = max.(logits, 0) - logits .* real + log.(1 + exp.(-abs.(logits)))
    loss = sum(loss)
    if average
        return loss / size(logits, 2)
    end
    return loss
end

function ganloss(atype, gforw, dforw)
    """
    Discriminator maximizes logD(x) + log(1-D(G(z))
    Generator maximizes log(D(G(z)))
    These are equivalent to minimizing nll
    logD(x) -> minimize nll between D(x) and 1
    log(1-D(G(z))) -> minimize nll between D(G(z)) and 0
    log(D(G(z))) -> minimize nll between D(G(z)) and 1
    """
    # TODO: gloss and dloss calls forward. Can we fix that?

    function gloss(gparams, dparams, x, z)
        println("Generator")
        batchsize = size(x)[end]
        positive = atype(ones(1, batchsize))

        generated = gforw(z, gparams) # G(z)
        fakeclss = reshape(dforw(generated, dparams), 1, batchsize) # D(G(z))
        return binaryxentropy(positive, fakeclss)
    end

    function dloss(dparams, gparams, x, z)
        println("Dis")
        batchsize = size(x)[end]
        negative = atype(zeros(1, batchsize))
        positive = atype(ones(1, batchsize))

        generated = gforw(z, gparams) # G(z)
        fakeclss = reshape(dforw(generated, dparams), 1, batchsize) # D(G(z))
        realclss = reshape(dforw(x, dparams), 1, batchsize) # D(x)

        return binaryxentropy(positive, realclss) + binaryxentropy(negative, fakeclss)
    end
    return gloss, dloss
end

function gangrad(atype, gforw, dforw)
    gloss, dloss = ganloss(atype, gforw, dforw)
    return gradloss(gloss), gradloss(dloss)
end
