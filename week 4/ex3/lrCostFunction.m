function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%----------------------NOTE---------------
%this is regularized cost function, the task statement is misleading in the pdf question

%lets first find h(theta)
g_x = X * theta;
h_theta = sigmoid(g_x);

temp_theta = theta;
temp_theta(1) = 0; %ignore theta_zero while summing the regularization term


J = -1 / m * sum(y .* log(h_theta) + (1-y) .* log(1 - h_theta)) + lambda / (2*m) * sum(temp_theta.^2);

%lets compute regularized gradient

error_diff = h_theta - y;

grad_1 = 1 / m * error_diff' * X(:,1);

temp_X = X(:, 2:end);

grad_rest = 1 / m * (error_diff' * temp_X + lambda * theta(2:end)');

grad = [grad_1 grad_rest]

% =============================================================

grad = grad(:);

end