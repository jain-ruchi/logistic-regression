function gradient = wgradient(x, y, b, w)
% compute the gradient with respect to w
    gradient = sigmoid(b .+ x * w);
    gradient = gradient .- y;
    gradient = transpose(x) * gradient;
    return
endfunction