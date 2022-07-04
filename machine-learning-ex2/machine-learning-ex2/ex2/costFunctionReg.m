function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J_temp=0;
for i=1:m
    J_temp = J_temp + ( -y(i) * log(sigmoid(X(i,:) * theta)) - (1 - y(i)) * log(1 - sigmoid(X(i,:) * theta)));
end
J = (1/m) * J_temp + (lambda / (2*m)) * sum (theta(2:size(theta)).^2);
grad = zeros(size(theta));

temp = 0;
for j =1:m
    temp = temp + (sigmoid(X(j,:) * theta) - y(j)) * X(j,1);
end
grad(1) = (1/m) * temp

for i = 2:size(theta)
    temp = 0;
    for j =1:m
        temp = temp + (sigmoid(X(j,:) * theta) - y(j)) * X(j,i);
    end
    grad(i) = (1/m) * temp +  (lambda / m) * theta(i);
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
