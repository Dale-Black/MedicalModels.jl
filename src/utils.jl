# 3D layer utilities
conv = (x, stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=SamePad())(x)
tran = (x, stride, in, out) -> ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=SamePad())(x)
norm = (x, channels) -> BatchNorm(channels)(x)

concat = (a, b) -> cat(a, b, dims=4)

conv1 = (x, in, out) -> leakyrelu.(norm(conv(x, 1, in, out), out))
conv2 = (x, in, out) -> leakyrelu.(norm(conv(x, 2, in, out), out))
tran2 = (x, in, out) -> leakyrelu.(norm(tran(x, 2, in, out), out))

# TODO: 2D layer utilities