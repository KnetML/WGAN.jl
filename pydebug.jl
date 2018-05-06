using PyCall
include("utils.jl")
include("models.jl")
include("loss.jl")
include("layers.jl")
@pyimport pickle

model = dcgan

function torch2knet(params)
    i = 1
    newparams = Any[]
    while i <= length(params) # This is only for convolutional discriminator
        name, param = params[i]
        if contains(name, "conv") || contains(name, "convt")  # Convert PyTorch filters to Knet filters e.g: (64, 3, 4, 4) -> (4, 4, 3, 64)
            push!(newparams, permutedims(param, [3, 4, 2, 1]))
        elseif contains(name, "batchnorm") # Batch norm params
            param_scale = convert(Array{Float32, 1}, param)
            param_bias = convert(Array{Float32, 1}, params[i+1][2])
            push!(newparams, vcat(param_scale, param_bias))
            i += 1
        else # MLP Params
            weights = convert(Array{Float32, 2}, param)
            bias = convert(Array{Float32, 1}, params[i+1][2])
            push!(newparams, weights)
            push!(newparams, bias)
            i += 1
        end
        i += 1
    end
    return newparams
end

function unpickle(path)
    pickle.load(pybuiltin(:open)(path, "rb"))
end

function gettorchparams(basedir)
    wDpath = joinpath(basedir, "wD.pk")
    wGpath = joinpath(basedir, "wG.pk")
    return unpickle(wDpath), unpickle(wGpath)
end

# Leaky relu test
inp = hcat([[1.8495, -0.8562, -0.5154], [-0.7537,  0.0551, -1.1757], [-0.5982,  1.4909, 0.8646]]...)
expected =  hcat([[1.8495, -0.1712, -0.1031], [-0.1507, 0.0551, -0.2351], [-0.1196, 1.4909, 0.8646]]...)
calculated = round.(leakyrelu.(inp, 0.2), 4)
diff = sum(expected - calculated)
println("LeakyReLU test: $diff") # This should always be 0

basedir = "/home/cem/WassersteinGAN/debug"

wD, wG = gettorchparams(basedir)
wD = torch2knet(wD)
wG = torch2knet(wG)

wD = KnetArray{Float32}.(wD)
wG = KnetArray{Float32}.(wG)

Dreal_in = permutedims(unpickle(joinpath(basedir, "Dreal_in.pk")), [3,4,2,1]) # (64, 3, 64, 64) -> (64, 64, 3, 64) BxCxHxW -> HxWxCxB
Dreal_out = round(unpickle(joinpath(basedir, "Dreal_out.pk"))[1], 4)

leak, zsize, clip = 0.2, 100, 0.01

generator, discriminator = model(leak, zsize, KnetArray{Float32})

_, gmoments, gforw = generator
_, dmoments, dforw = discriminator

for i = 1:length(wD)
    wD[i] = clipfun(wD[i], -clip, clip)
end

Dreal_in = KnetArray{Float32}(Dreal_in)
knetDout_real = mean(dforw(wD, dmoments, Dreal_in, leak))
knetDout_real = round(knetDout_real, 4)
println("Knet Dout Real $knetDout_real, PyTorch Dout Real $Dreal_out, Diff: ", abs(round(knetDout_real-Dreal_out, 4)))

Dfake_noise = permutedims(unpickle(joinpath(basedir, "Dfake_noise.pk")), [3,4,2,1]) # (64, 100, 1, 1) -> (1, 1, 100, 64) BxCxHxW -> HxWxCxB
Dfake_out = round(unpickle(joinpath(basedir, "Dfake_out.pk"))[1], 4)

Dfake_noise = KnetArray{Float32}(Dfake_noise)
generated = gforw(wG, gmoments, Dfake_noise)

Gout = permutedims(unpickle(joinpath(basedir, "Gout.pk")), [3,4,2,1]) # (64, 3, 64, 64) -> (64, 64, 3, 64)
knetGout = round(vecnorm(generated), 4)
pyGout = round(vecnorm(Gout), 4)
println("Knet Gout $knetGout, PyTorch Gout $pyGout, Diff: ", abs(round(knetGout-pyGout, 4)))

knetDout_fake = mean(dforw(wD, dmoments, generated, leak))
knetDout_fake = round(knetDout_fake, 4)
println("Knet Dout Fake $knetDout_fake, PyTorch Dout Fake $Dfake_out, Diff: ", abs(round(knetDout_fake-Dfake_out,4)))

knetDerr = round(knetDout_real - knetDout_fake, 4)
pyDerr = round(Dreal_out - Dfake_out, 4)
println("Knet Dloss $knetDerr, PyTorch Dloss $pyDerr, Diff: ", abs(round(knetDerr-pyDerr, 4)))

# End of testing forward passes

# Test training Discriminator
# Reinit moments
dopts = optimizers(wD, Rmsprop, lr=0.00005, rho=0.99, eps=1e-8)

generator, discriminator = model(leak, zsize, KnetArray{Float32})

_, gmoments, gforw = generator
_, dmoments, dforw = discriminator

generated = gforw(wG, gmoments, Dfake_noise) # Pass again for moments

dgrad, dl = wganDgradloss2(wD, dmoments, dforw, Dreal_in, generated, leak)

wD_grad = unpickle(joinpath(basedir, "Dparam_grad.pk"))
wD_grad = torch2knet(wD_grad)

Dgraddiff = round(mean(abs.(vecnorm.(wD_grad) - vecnorm.(dgrad))), 4)
println("D Grad diff: $Dgraddiff")

update!(wD, dgrad, dopts)
wD_updated = torch2knet(unpickle(joinpath(basedir, "wD_updated.pk")))
Dupdatediff = round(mean(abs.(vecnorm.(wD_updated) - vecnorm.(wD))), 4)
println("D Updated diff: $Dupdatediff")

wD = KnetArray{Float32}.(wD_updated)

# We have trained discriminator for one step. Test training of generator after
# the discriminator training step.
gopts = optimizers(wG, Rmsprop, lr=0.00005, rho=0.99, eps=1e-8)
z = permutedims(unpickle(joinpath(basedir, "G_noise.pk")), [3,4,2,1])
z = KnetArray{Float32}(z)
fake = gforw(wG, gmoments, z)

Gout = permutedims(unpickle(joinpath(basedir, "G_generated.pk")), [3,4,2,1]) # (64, 3, 64, 64) -> (64, 64, 3, 64)
Gdiff = round(abs(vecnorm(fake) - vecnorm(Gout)), 4)
println("G generated diff: $Gdiff")

ggrad, gloss = wganGgradloss(wG, wD, gmoments, dmoments, gforw, dforw, z, leak)
gloss = round(gloss, 4)

pyGloss = -round(unpickle(joinpath(basedir, "G_loss.pk"))[1], 4)
pyGgrad = unpickle(joinpath(basedir, "G_grad.pk"))
wG_updated = torch2knet(unpickle(joinpath(basedir, "wG_updated.pk")))
pyGgrad = torch2knet(pyGgrad)

Gloss_diff = round(abs(pyGloss-gloss), 4)
Ggrad_diff = round(mean(abs.(vecnorm.(ggrad) - vecnorm.(pyGgrad))), 4)
update!(wG, ggrad, gopts)
Gupdate_diff = round(mean(abs.(vecnorm.(wG_updated) - vecnorm.(wG))), 4)

println("Knet Gloss $gloss, PyTorch Gloss $pyGloss, Diff: $Gloss_diff")
println("G Grad diff: $Ggrad_diff")
println("G update diff: $Gupdate_diff")
