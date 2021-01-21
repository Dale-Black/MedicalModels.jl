include("./imports.jl")

@testset ExtendedTestSet "Unet2D" begin
    @testset ExtendedTestSet "Unet2D" begin
    model = Unet2D(1, 1)
	x = rand(Float32, 64, 64, 1, 1)
	x̂ = model(x)
    @test size(x) == size(x̂)
    end

    @testset ExtendedTestSet "Unet2D" begin
    model = Unet2D(3, 1)
	x = rand(Float32, 64, 64, 3, 1)
	x̂ = model(x)
    @test size(x) != size(x̂)
    end

    @testset ExtendedTestSet "Unet2D" begin
    model = Unet2D(3, 1)
	x = rand(Float32, 64, 64, 3, 1)
	x̂ = model(x)
    @test [size(x, 1), size(x, 2)] == [size(x̂, 1), size(x̂, 2)]
    end
end