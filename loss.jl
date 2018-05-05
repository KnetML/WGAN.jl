using Knet

function clipfun(param, minval, maxval)
    KnetArray{Float32}(clamp.(Array{Float32}(param), minval, maxval))
end

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

function ganDloss(dparams, dmoments, dforw, real, fake, positive, negative, leak)
    realclss = mat(dforw(dparams, dmoments, real, leak))
    fakeclss = mat(dforw(dparams, dmoments, fake, leak))
    return binaryxentropy(realclss, positive) + binaryxentropy(fakeclss, negative)
end

function wganDloss(dparams, dmoments, dforw, input, leak)
    clss = dforw(dparams, dmoments, input, leak)
    return mean(clss)
end

function wganDloss2(dparams, dmoments, dforw, real, fake, leak)
    realclss = dforw(dparams, dmoments, real, leak)
    fakeclss = dforw(dparams, dmoments, fake, leak)
    return -(mean(realclss) - mean(fakeclss))
end

function ganGloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, positive, leak)
    fakeimg = gforw(gparams, gmoments, z)
    fakeclss = mat(dforw(dparams, dmoments, fakeimg, leak))
    return binaryxentropy(fakeclss, positive)
end

function wganGloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, leak)
    fakeimg = gforw(gparams, gmoments, z)
    fakeclss = dforw(dparams, dmoments, fakeimg, leak)
    return -mean(fakeclss)
end

# Gradient functions
ganGgradloss = gradloss(ganGloss)
ganDgradloss = gradloss(ganDloss)
wganGgradloss = gradloss(wganGloss)
wganDgradloss = gradloss(wganDloss)
wganDgradloss2 = gradloss(wganDloss2) # 2 means combining both losses

function traingan(zsize, atype, metric, clip)
    """
    zsize: Noise size
    atype: Array type
    metric: wgan or gan, if wasserstein, input x has n many elements where
    n is sampled minibatches for training D more. In addition if metric is wasserstein
    clip must be provided.
    """
    metric = lowercase(metric)

    function trainD(dparams, gparams, gmoments, dmoments, gforw, dforw, x, opts, leak)
        batchsize = size(x)[end]
        z = samplenoise4(zsize, batchsize, atype)
        generated = gforw(gparams, gmoments, z)

        if metric == "wgan"
            for i = 1:length(dparams)
                dparams[i] = clipfun(dparams[i], -clip, clip)
            end
            grad, loss = wganDgradloss2(dparams, dmoments, dforw, x, generated, leak)
            # gradcheck(wganDloss, dparams, dmoments, dforw, generated, x, leak, atol=0.1, verbose=true, gcheck=100)
        elseif metric == "gan"
            positive = atype(ones(1, batchsize))
            negative = atype(zeros(1, batchsize))
            grad, loss = ganDgradloss(dparams, dmoments, dforw, x, generated, positive, negative, leak)
        else
            throw(ArgumentError("Unknown metric"))
        end

        update!(dparams, grad, opts)

        return loss
    end

    function trainG(gparams, dparams, gmoments, dmoments, gforw, dforw, batchsize, opts, leak)
        positive = atype(ones(1, batchsize))
        z = samplenoise4(zsize, batchsize, atype)

        if metric == "wgan"
            grad, loss = wganGgradloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, leak)
            # gradcheck(wganGloss, gparams, dparams, gmoments, dmoments, gforw, dforw, z, leak, atol=0.01, verbose=true)
        elseif metric == "gan"
            grad, loss = ganGgradloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, positive, leak)
        else
            throw(ArgumentError("Unknown metric"))
        end

        update!(gparams, grad, opts)

        return loss
    end

    return trainD, trainG
end
