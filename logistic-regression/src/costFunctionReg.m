function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

	% Initialize some useful values
	m = length(y); % number of training examples
	J = 0;
	grad = zeros(size(theta));
	
	% Compute Cost (J) of a particular choice of theta. 
	% Compute the partial derivatives and set grad to the partial derivatives of the cost w.r.t. each parameter in theta 
	regularizedAddon=(lambda/(2*m))*(sum(theta .^ 2)  - (theta(1,1)^2));
	hypothesis=sigmoid(X* theta);
	J= (((-1) * y' * log(hypothesis)) - ((1 .- y)' *log( 1 .- hypothesis)))/m + regularizedAddon;
	diff = hypothesis - y;
	grad= 1/m *(diff' * X )' ;

	 for i=2:length(grad),
		grad(i,1)=grad(i,1)+ (lambda/m)*theta(i,1);
	end
end