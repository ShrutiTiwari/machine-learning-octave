function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

	% Initialize some useful values
	m = length(y); % number of training examples
	J = 0;
	grad = zeros(size(theta));
	
	% Compute Cost (J) of a particular choice of theta. 
	% Compute the partial derivatives and set grad to the partial derivatives of the cost w.r.t. each parameter in theta 
	
	hypothesis = sigmoid(X* theta);
	J = (((-1) * y' * log(hypothesis)) - ((1 .- y)' *log( 1 .- hypothesis)))/m;
	diff = hypothesis - y;
	grad = 1/m *(diff' * X )';
end
