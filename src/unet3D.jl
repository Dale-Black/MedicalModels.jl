function unet3D(x, in_chs, lbl_chs)
    # Contracting Layers
    l1 = conv1(x, in_chs, in_chs*2)
    l2 = conv1(conv2(l1, in_chs*2, in_chs*4), in_chs*4, in_chs*4)
    l3 = conv1(conv2(l2, in_chs*4, in_chs*8), in_chs*8, in_chs*8)
    l4 = conv1(conv2(l3, in_chs*8, in_chs*16), in_chs*16, in_chs*16)
    l5 = conv1(conv2(l4, in_chs*16, in_chs*32), in_chs*32, in_chs*32)

    # Exapanding Layers
    l6 = conv1(concat(tran2(l5, in_chs*32, in_chs*16), l4), in_chs*32, in_chs*16)
    l7 = conv1(concat(tran2(l6, in_chs*16, in_chs*8), l3), in_chs*16, in_chs*8)
    l8 = conv1(concat(tran2(l7, in_chs*8, in_chs*4), l2), in_chs*8, in_chs*4)
    l9 = conv1(concat(tran2(l8, in_chs*4, in_chs*2), l1), in_chs*4, in_chs*2)
    l10 = conv1(l9, in_chs*2, lbl_chs)
end