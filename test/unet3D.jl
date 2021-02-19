include("./imports.jl")

@testset ExtendedTestSet "unet3D" begin
    @testset ExtendedTestSet "unet3D" begin
	x = rand(Float32, 96, 96, 96, 1, 1)
    x̂ = unet3D(x, 1, 1)
    @test size(x) == size(x̂)
    end

    @testset ExtendedTestSet "unet3D" begin
	x = rand(Float32, 64, 64, 64, 3, 1)
    x̂ = unet3D(x, 3, 1)
    @test size(x) != size(x̂)
    end

    @testset ExtendedTestSet "unet3D" begin
	x = rand(Float32, 64, 64, 64, 3, 1)
    x̂ = unet3D(x, 3, 1)
    @test [size(x, 1), size(x, 2), size(x, 3)] == [size(x̂, 1), size(x̂, 2), size(x̂, 3)]
    end
end