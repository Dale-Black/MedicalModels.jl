module MedicalModels

using Flux
using Functors

include("./Unet3D.jl")
include("./utils.jl")

export Unet3D,
    Unet2D

end
