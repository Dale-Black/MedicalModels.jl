using MedicalModels
using Test

@testset ExtendedTestSet "Unet2D" begin
    @testset ExtendedTestSet "Unet2D" begin
    model = Unet2D(1, 1)
	x = rand(Float32, 64, 64, 1, 1)
	x̂ = model(x)
    @test size(x) == size(x̂)
    end
end