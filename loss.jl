using Knet

function binaryxentropy(logits, gold; average=true)
    """
    Sigmoid Binary Cross Entropy Loss
    """
    loss = max.(logits, 0) - logits .* gold + log.(exp.(abs.(logits) .* -1) .+ 1)
    loss = sum(loss)
    if average
        return loss / size(logits, 2)
    end
    return loss
end

function ganDloss(dparams, dmoments, dforw, real, fake, positive, negative, leak, metric)
    realclss = mat(dforw(dparams, dmoments, real, leak))
    fakeclss = mat(dforw(dparams, dmoments, fake, leak))
    if metric == "gan"
        return binaryxentropy(realclss, positive) + binaryxentropy(fakeclss, negative)
    else
        return mean(realclss, 2)[1][1] - mean(fakeclss, 2)[1][1]
    end
end

function ganGloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, positive, leak, metric)
    fakeimg = gforw(gparams, gmoments, z)
    fakeclss = mat(dforw(dparams, dmoments, fakeimg, leak))
    if metric == "gan"
        return binaryxentropy(fakeclss, positive)
    else
        return mean(fakeclss, 2)[1][1]
    end
end

# Gradient functions
ganGgradloss = gradloss(ganGloss)
ganDgradloss = gradloss(ganDloss)

function traingan(zsize, atype, metric, clip)
    """
    zsize: Noise size
    atype: Array type
    metric: wgan or gan, if wasserstein, input x has n many elements where
    n is sampled minibatches for training D more. In addition if metric is wasserstein
    clip must be provided.
    """
    metric = lowercase(metric)

    if metric == "wgan"
        @assert clip != nothing
        clipfun = clipper(clip, atype)
    end

    function trainD(dparams, gparams, gmoments, dmoments, gforw, dforw, x, opts, leak)
        batchsize = size(x)[end]

        positive = atype(ones(1, batchsize))
        negative = atype(zeros(1, batchsize))
        z = samplenoise4(zsize, batchsize, atype)

        generated = gforw(gparams, gmoments, z)
        grad, loss = ganDgradloss(dparams, dmoments, dforw, x, generated, positive, negative, leak, metric)
        update!(dparams, grad, opts)

        metric == "wgan" && map!(clipfun, dparams, dparams)
        
        return loss
    end

    function trainG(gparams, dparams, gmoments, dmoments, gforw, dforw, batchsize, opts, leak)
        positive = atype(ones(1, batchsize))
        negative = atype(zeros(1, batchsize))
        z = samplenoise4(zsize, batchsize, atype)

        grad, loss = ganGgradloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, positive, leak, metric)
        metric == "wgan" && grad *= -1
        update!(gparams, grad, opts)

        return loss
    end
    return trainD, trainG
end

function clipper(clip, atype)
    function clipfun(param)
        if atype == KnetArray{Float32}
            param = Array{Float32}(param)
        end
        return atype(clamp.(param, -clip, clip))
    end
    return clipfun
end
