# Adapted from https://github.com/DhairyaLGandhi/UNet.jl/blob/master/src/model.jl

function BatchNormWrap3D(out_ch)
  Chain(x -> expand_dims(x, 3), BatchNorm(out_ch), x -> squeeze(x))
end

UNetConvBlock3D(in_chs, out_chs, kernel = (3, 3, 3)) = 
	Chain(
		Conv(kernel, in_chs => out_chs, pad = (1, 1, 1)), 
		BatchNormWrap3D(out_chs),
    x -> leakyrelu.(x, 0.2f0))
    
ConvDown3D(in_chs, out_chs, kernel = (4, 4, 4)) = Chain(
  Conv(kernel, in_chs => out_chs, pad = (1, 1, 1), stride = (2, 2, 2)),
  BatchNormWrap3D(out_chs),
  x -> leakyrelu.(x, 0.2f0))

	struct UNetUpBlock3D
    upsample
end

@functor UNetUpBlock3D

UNetUpBlock3D(in_chs::Int, out_chs::Int; kernel = (3, 3, 3), p = 0.5f0) = 
  UNetUpBlock3D(
    Chain(
      x -> leakyrelu.(x, 0.2f0),
      ConvTranspose((2, 2, 2), in_chs => out_chs, stride = (2, 2, 2)),
      BatchNormWrap3D(out_chs),
      Dropout(p)))

function (u::UNetUpBlock3D)(x, bridge)
  x = u.upsample(x)
  return cat(x, bridge, dims = 4)
end

struct Unet3D
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor Unet3D

function Unet3D(channels::Int = 1, labels::Int = channels)
  conv_down_blocks = Chain(
    ConvDown3D(64, 64),
    ConvDown3D(128, 128),
    ConvDown3D(256, 256),
    ConvDown3D(512, 512))

  conv_blocks = Chain(
    UNetConvBlock3D(channels, 3),
    UNetConvBlock3D(3, 64),
    UNetConvBlock3D(64, 128),
    UNetConvBlock3D(128, 256),
    UNetConvBlock3D(256, 512),
    UNetConvBlock3D(512, 1024),
    UNetConvBlock3D(1024, 1024))

  up_blocks = Chain(
    UNetUpBlock3D(1024, 512),
    UNetUpBlock3D(1024, 256),
    UNetUpBlock3D(512, 128),
    UNetUpBlock3D(256, 64,p = 0.0f0),
    Chain(x -> leakyrelu.(x, 0.2f0),
    Conv((1, 1, 1), 128 => labels)))

  Unet(conv_down_blocks, conv_blocks, up_blocks)
  end

function (u::Unet3D)(x::AbstractArray)
  op = u.conv_blocks[1:2](x)

  x1 = u.conv_blocks[3](u.conv_down_blocks[1](op))
  x2 = u.conv_blocks[4](u.conv_down_blocks[2](x1))
  x3 = u.conv_blocks[5](u.conv_down_blocks[3](x2))
  x4 = u.conv_blocks[6](u.conv_down_blocks[4](x3))

  up_x4 = u.conv_blocks[7](x4)

  up_x1 = u.up_blocks[1](up_x4, x3)
  up_x2 = u.up_blocks[2](up_x1, x2)
  up_x3 = u.up_blocks[3](up_x2, x1)
  up_x5 = u.up_blocks[4](up_x3, op)
  tanh.(u.up_blocks[end](up_x5))
end