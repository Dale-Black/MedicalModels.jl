module MedicalModels

using Flux
using Flux: Parallel

include("./unet3D.jl")
include("./utils.jl")

export 
    # Export unet3D.jl functions
    unet3D,

    # Export utils.jl functions
    conv,
    tran,
    conv1,
    conv2,
    tran2

end
