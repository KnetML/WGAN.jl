include("utils.jl")
using ArgParse

function main(args)
    s = ArgParseSettings()
    s.description = "LSUN Bedroom dataset preprocessing."

    @add_arg_table s begin
        ("--path"; arg_type=String; required=true; help="Path of raw dataset")
        ("--out"; arg_type=String; required=true; help="Output path")
        ("--bsize"; arg_type=Int; required=true; help="Divide dataset into batches with size bsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)

    processimgs(o[:path], o[:bsize], o[:out])
end

main("--path /home/cem/bedroom_train --out /home/cem/bedroom --bsize 10000")
# @time imgs = normalize(loadimgtensors("/home/cem/bedroom", (1,10)), 0, 1)
# println(size(imgs))
#saveimgs(imgs[1:40000,:,:,:])
