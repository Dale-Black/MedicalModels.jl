using MedicalModels
using Test

@testset ExtendedTestSet "unet" begin
    @testset ExtendedTestSet "unet" begin
    model = Unet(1, 1)
	x = rand(Float32, 64, 64, 64, 1, 1)
	x̂ = model(x)
    @test size(x) == size(x̂)
    end
end