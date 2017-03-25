function gradient = bgradient(x, y, b, w)
% compute the gradient with respect to b
    sig = sigmoid(b .+ x * w);
    gradient = sum(sig .- y);
    return
endfunction