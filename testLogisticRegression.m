function accuracy = testLogisticRegression(x, y, w, b)
% tests accuracy of trained logistic model
% Input:
%   x: feature matrix N x M
%   y: labels N x 1
%   w: weight vector M x 1
%   b: bias term
%
% Output:
%   accuracy: average accuracy

sig = sigmoid(b .+ x * w);
predictions = round(sig);
accu_sum = sum(predictions == y);
accuracy = accu_sum / rows(y);

endfunction