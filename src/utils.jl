expand_dims(x, n::Int) = reshape(x, ones(Int64,n)..., size(x)...)

function squeeze(x)
	if size(x)[end] == 1 && size(x)[end-1] != 1
		# For the case BATCH_SIZE = 1 and Channels != 1
		int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)..., 1)
	elseif size(x)[end] != 1 && size(x)[end-1] == 1
		# For the case BATCH_SIZE != 1 and Channels = 1
		int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)..., 1, :)
	elseif size(x)[end] == 1 && size(x)[end-1] == 1
		# For the case BATCH_SIZE = 1 and Channels = 1
		int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)..., 1, 1)
	else
		size(x)[end] != 1 && size(x)[end-1] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    end
end