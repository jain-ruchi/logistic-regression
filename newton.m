function [w, b] = newton(x, y, w, b, lambda, iter)
% apply Newtons Method to weights
% Input:
%   x: feature matrix N x M
%   y: labels N x 1
%   w: initialized w
%   b: initialized b
%   step_size: step size
%   lambda: regularization coefficient
%   iter: number of iterations to run
%
% Output:
%   w: weight vector M x 1
%   b: bias term 1 x 1

    identity_mat = eye(columns(x));

    for j = 1:iter
        temp_w = w - pinv(whessian(x, b, w) + 2 * identity_mat * lambda) * (wgradient(x, y, b, w) + 2 * lambda * w);
        temp_b = b - pinv(bhessian(x, b, w)) * bgradient(x, y, b, w);
        w = temp_w;
        b = temp_b;
    endfor
endfunction

function hessian = bhessian(x, b, w)
% compute hessian with respect to b
    sig = sigmoid(b .+ x * w);
    hessian = sum(sig .* (1 - sig));
    return
endfunction

function hessian = whessian(x, b, w)
% compute hessian with respect to w
    sig = sigmoid(b .+ x * w);
    sig = sig .* (1 - sig);
    hessian = zeros(columns(x), columns(x));

    for i = 1:rows(x)
        hessian = hessian + sig(i) .* (transpose(x(i,:)) .* x(i,:));
    endfor
    return
endfunction