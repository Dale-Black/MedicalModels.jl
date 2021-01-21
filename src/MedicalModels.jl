module MedicalModels

using Flux
using Functors

include("./unet3D.jl")
include("./unet2D.jl")
include("./utils.jl")

export Unet3D,
    Unet2D

end
