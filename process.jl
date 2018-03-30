include("utils.jl")
using ArgParse, Logging

Logging.configure(output=open("process.log", "w"))
Logging.configure(level=INFO)

function main(args)
    s = ArgParseSettings()
    s.description = "LSUN Bedroom dataset preprocessing."

    @add_arg_table s begin
        ("--path"; arg_type=String; required=true; help="Path of raw dataset")
        ("--n"; arg_type=Int; required=true; help="How many images to read. -1 means read everything")
        ("--out"; arg_type=String; required=true; help="Output path")
        ("--bsize"; arg_type=Int; required=true; help="Divide output into batches with size bsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)

    info("Reading dataset")
    @time myimgs = readimgs(o[:path], o[:n])
    info("Dataset size: ",size(myimgs))

    info("Saving image tensors")
    saveimgtensors(o[:out], myimgs, o[:bsize])
    info("Done")
end

main("--path /home/cem/bedroom_train --n -1 --out /home/cem/bedroom --bsize 10000")
# imgs = loadimgtensors("/home/cem/bedroom")
# saveimgs(imgs)
