using Knet

function binaryxentropy(logits, gold; average=true)
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

function ganGloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, positive, leak)
    fakeimg = gforw(gparams, gmoments, z)
    fakeclss = mat(dforw(dparams, dmoments, fakeimg, leak))
    return binaryxentropy(fakeclss, positive)
end

ganGgradloss = gradloss(ganGloss)
ganDgradloss = gradloss(ganDloss)

function traingan(zsize, atype)
    function trainD(dparams, gparams, gmoments, dmoments, gforw, dforw, x, opts, leak)
        batchsize = size(x)[end]

        positive = atype(ones(1, batchsize))
        negative = atype(zeros(1, batchsize))
        z = samplenoise4(zsize, batchsize, atype)

        generated = gforw(gparams, gmoments, z)
        grad, loss = ganDgradloss(dparams, dmoments, dforw, x, generated, positive, negative, leak)
        update!(dparams, grad, opts)
        return loss
    end

    function trainG(gparams, dparams, gmoments, dmoments, gforw, dforw, x, opts, leak)
        batchsize = size(x)[end]

        positive = atype(ones(1, batchsize))
        negative = atype(zeros(1, batchsize))
        z = samplenoise4(zsize, batchsize, atype)

        grad, loss = ganGgradloss(gparams, dparams, gmoments, dmoments, gforw, dforw, z, positive, leak)
        update!(gparams, grad, opts)
        return loss
    end

    return trainD, trainG
end
