using PyCall
include("utils.jl")
@pyimport torchvision.transforms as transforms
@pyimport torchvision.datasets as dset
@pyimport torch.utils.data as d
@pyimport torch

function getdataset(dbpath)
    dataset = dset.LSUN(db_path=dbpath, classes=["bedroom_train"],
                    transform=transforms.Compose([transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    return dataset
end

function getdataiter(dataset, batchsize)
    dataloader = d.DataLoader(dataset, batch_size=batchsize, shuffle=true, num_workers=2)
    return pybuiltin(:iter)(dataloader)
end

function getnext(dataiter)
    return tojlarr(pybuiltin(:next)(dataiter))
end

function tojlarr(tensor)
    img = tensor[1]
    nptensor = img[:numpy]() # Call's torch.FloatTensor's numpy() method.
    jltensor = convert(Array{Float32}, nptensor) # BxCxHxW
    jltensor = permutedims(jltensor, [1,3,4,2]) # Channel last BxHxWxC
    jltensor = permutedims(jltensor, [2,3,4,1]) # Batch last
    return jltensor
end
