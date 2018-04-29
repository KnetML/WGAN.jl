using Knet

function clipfun(x0, minval, maxval)
    x1 = max.(minval, x0)
    x2 = min.(maxval, x1)
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

function wganDloss_real(dparams, dmoments, dforw, real, leak)
    realclss = mat(dforw(dparams, dmoments, real, leak))
    return mean(realclss)
end

function wganDloss_fake(dparams, dmoments, dforw, fake, leak)
    fakeclss = mat(dforw(dparams, dmoments, fake, leak))
    return mean(fakeclss)
end

function ganGloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, positive, leak)
    fakeimg = gforw(gparams, gmoments, z)
    fakeclss = mat(dforw(dparams, dmoments, fakeimg, leak))
    return binaryxentropy(fakeclss, positive)
end

function wganGloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, leak)
    fakeimg = gforw(gparams, gmoments, z)
    fakeclss = mat(dforw(dparams, dmoments, fakeimg, leak))
    return mean(fakeclss)
end

# Gradient functions
ganGgradloss = gradloss(ganGloss)
ganDgradloss = gradloss(ganDloss)
wganGgradloss = gradloss(wganGloss)
wganDgradloss_fake = gradloss(wganDloss_fake)
wganDgradloss_real = gradloss(wganDloss_real)

function traingan(zsize, atype, metric, clip)
    """
    zsize: Noise size
    atype: Array type
    metric: wgan or gan, if wasserstein, input x has n many elements where
    n is sampled minibatches for training D more. In addition if metric is wasserstein
    clip must be provided.
    """
    metric = lowercase(metric)

    function trainD(dparams, gparams, gmoments, dmoments, gforw, dforw, x, opts, leak, batchsize)
        # batchsize = size(x)[end]
        z = samplenoise4(zsize, batchsize, atype)
        generated = gforw(gparams, gmoments, z)

        if metric == "wgan"
            for i = 1:length(dparams)
                dparams[i] = clipfun(dparams[i], -clip, clip)
            end

            grad_real, loss_real = wganDgradloss_real(dparams, dmoments, dforw, x, leak)
            grad_fake, loss_fake = wganDgradloss_fake(dparams, dmoments, dforw, generated, leak)

            grad = grad_real - grad_fake
            loss = loss_real - loss_fake
        elseif metric == "gan"
            positive = atype(ones(1, batchsize))
            negative = atype(zeros(1, batchsize))
            grad, loss = ganDgradloss(dparams, dmoments, dforw, x, generated, positive, negative, leak)
        else
            throw(ArgumentError("Unknown metric"))
        end
        # println("D grad mean")
        # println(mean([mean(k) for k in grad]))
        update!(dparams, grad, opts)
        return loss
    end

    function trainG(gparams, dparams, gmoments, dmoments, gforw, dforw, batchsize, opts, leak)
        positive = atype(ones(1, batchsize))
        z = samplenoise4(zsize, batchsize, atype)

        if metric == "wgan"
            grad, loss = wganGgradloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, leak)
        elseif metric == "gan"
            grad, loss = ganGgradloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, positive, leak)
        else
            throw(ArgumentError("Unknown metric"))
        end
        # println("G Grad")
        #gtemp = deepcopy(gparams)
        #println(length(gparams), length(grad))
        # println(mean([mean(k) for k in grad]))
        update!(gparams, grad, opts)
        #println([mean(abs.(k)) for k in gtemp-gparams])
        return loss
    end

    return trainD, trainG
end
