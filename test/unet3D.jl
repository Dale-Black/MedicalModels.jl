include("./imports.jl")

@testset ExtendedTestSet "Unet3D" begin
    @testset ExtendedTestSet "Unet3D" begin
    model = Unet3D(1, 1)
	x = rand(Float32, 64, 64, 64, 1, 1)
	x̂ = model(x)
    @test size(x) == size(x̂)
    end

    @testset ExtendedTestSet "Unet3D" begin
    model = Unet3D(3, 1)
	x = rand(Float32, 64, 64, 64, 3, 1)
	x̂ = model(x)
    @test size(x) != size(x̂)
    end

    @testset ExtendedTestSet "Unet3D" begin
    model = Unet3D(3, 1)
	x = rand(Float32, 64, 64, 64, 3, 1)
	x̂ = model(x)
    @test [size(x, 1), size(x, 2), size(x, 3)] == [size(x̂, 1), size(x̂, 2), size(x̂, 3)]
    end
end