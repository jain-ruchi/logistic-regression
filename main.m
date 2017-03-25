% PART 0: Parse the dataset

% get training and test data points
train_data = dlmread('ionosphere/ionosphere_train.dat', ',');
train_data(:,35) = [];
test_data = dlmread('ionosphere/ionosphere_test.dat', ',');
test_data(:,35) = [];

% map labels to numerical binary values
% b=bad: 1 and g=good: 0

train_label = zeros(rows(train_data),1);
test_label = zeros(rows(test_data),1);

fid = fopen('ionosphere/ionosphere_train.dat');
for i = 1:rows(train_data)
    txt = fgetl(fid);
    words = strsplit(txt, ',');

    if strcmp(words{35}, 'b')
        train_label(i) = 1;
    else
        train_label(i) = 0;
    endif
endfor
fclose(fid);

fid = fopen('ionosphere/ionosphere_test.dat');
for i = 1:rows(test_data)
    txt = fgetl(fid);
    words = strsplit(txt, ',');

    if strcmp(words{35}, 'b')
        test_label(i) = 1;
    else
        test_label(i) = 0;
    endif
endfor
fclose(fid);

% PART 1: Batch Gradient Descent

iter = 100;
step_sizes = [0.001; 0.01; 0.05; 0.1; 0.5];
lambdas = [0; 0.05; 0.1; 0.15; 0.2; 0.25; 0.3; 0.35; 0.4; 0.45; 0.5];

for k = 1:rows(step_sizes)
    for i = 1:rows(lambdas)

        [w, b] = gradient_descent(train_data, train_label, step_sizes(k), lambdas(i), iter);
        accuracy = testLogisticRegression(test_data, test_label, w, b);
        printf("Accuracy (Step Size=%d Lambda=%d): %d\n", step_sizes(k), lambdas(i), accuracy);
    endfor
endfor

% PART 2: Newtons Method

% Initialize w, b using gradient descent with five iterations
[newtown_w, newton_b] = gradient_descent(train_data, train_label, step_sizes(2), lambdas(2), 5);

for i = 1:rows(lambdas)
    w = newtown_w;
    b = newton_b;

    [w, b] = newton(train_data, train_label, w, b, lambdas(i), iter);
    accuracy = testLogisticRegression(test_data, test_label, w, b);
    printf("Accuracy (Lambda=%d): %d\n", lambdas(i), accuracy);
endfor
