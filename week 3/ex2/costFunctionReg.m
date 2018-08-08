function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
temp_n = size(theta);
n = temp_n(1,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
temp_theta = theta(2:n,:);

h_theta = sigmoid(X*theta);
J = -1 / m * ((y' * log(h_theta)) + (1 - y)' * log(1 - h_theta)) + lambda / (2 * m) * sum(temp_theta .^ 2);

temp1 = 1 / m * sum((h_theta - y) .* X(:,1));  %element wise multiplication of matrices

temp_x = X(:, 2:n);  %except first column


temp2 = 1 / m * ((h_theta - y)' * temp_x) + lambda / m * temp_theta';

grad = [temp1 temp2];

% =============================================================

end
