function result = sigmoid(z)
% Compute the sigmoid for given z value

result = 1 ./ (1 + e.^((-1) .* z));

result(result<1e-16) = 1e-16;
result(result>1e16) = 1e16;

return
endfunction