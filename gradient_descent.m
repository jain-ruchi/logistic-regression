function [w, b] = gradient_descent(x, y, step_size, lambda, iter)
% apply the gradient descent algorithm to weights
% Input:
%   x: feature matrix N x M
%   y: labels N x 1
%   step_size: step size
%   lambda: regularization coefficient
%   iter: number of iterations to run
%
% Output:
%   w: weight vector M x 1
%   b: bias term 1 x 1

    step_size = step_size * (1/rows(x));    % step size absorbs dataset size
    w = zeros(columns(x), 1);               % initialize weights to zeros
    b = 0.1;                                % initialize the offset to .1

    for j = 1:iter
        temp_w = w - step_size * (wgradient(x, y, b, w) + 2 * lambda * w);
        temp_b = b - step_size * bgradient(x, y, b, w);
        w = temp_w;
        b = temp_b;
    endfor

    return
endfunction
