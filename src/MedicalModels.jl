module MedicalModels

using Flux
using Functors

include("./unet.jl")
include("./utils.jl")

export Unet

end
