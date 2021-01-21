using MedicalModels
using Test

@testset ExtendedTestSet "Unet3D" begin
    @testset ExtendedTestSet "Unet3D" begin
    model = Unet3D(1, 1)
	x = rand(Float32, 64, 64, 64, 1, 1)
	x̂ = model(x)
    @test size(x) == size(x̂)
    end
end