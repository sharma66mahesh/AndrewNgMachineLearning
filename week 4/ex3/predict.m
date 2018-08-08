function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add x0 to matrix X
X = [ones(m, 1) X];

% Theta1 -- no. of rows-> no. of neurons in 2nd layer excluding bias unit
% Theta1 -- no. of cols-> no. of neurons in 1st layer

g_x1 = sigmoid(Theta1 * X');

% g_x has dimension K2 * m

% [~, a_1] = max(g_x, [], 1); %col-wise max

a_1 = [ones(m,1)'; g_x1];  % added a bias unit in first row

g_x2 = sigmoid(Theta2 * a_1);

[~, p] = max(g_x2, [], 1); % col-wise max

% =========================================================================


end
