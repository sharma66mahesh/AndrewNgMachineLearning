function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

%p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% we have value of theta. Now we compute h_theta. Get its sigmoid value. And ...
% select the max among the h_thetas of all the classes.

% X dimension:: m * n+1
% all_theta dimension:: K * n+1

h_theta = sigmoid(all_theta * X');

% h_theta dimension:: K * m

[~, p] = max(h_theta, [], 1);  %compute column-wise maximum. for row-wise max...
% write max(h_theta, [], 2); 
% [~,p] is equivalent to saying [dummy, p].\n Discard dummy.

% real ma vannu parda, hami lai max p ko value vanda pani max p ko index tha ...
% paye chai kun class ho vanera chuttauna sakinthyo hai

% =========================================================================


end
