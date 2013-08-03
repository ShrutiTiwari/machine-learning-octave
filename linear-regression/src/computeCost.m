function J = computeCost(X, y, theta)
%COSTFUNCTION computes the cost of using theta as the parameter for linear regression to fit the data points in X and y
%	Initialize some useful values
	m = length(y); % number of training examples
	J = 0;
	% Compute the cost(J) of a particular choice of theta
	d = y - (X * theta) ;
	C = d.^2;
	J = (sum(C))/(2*m);
end
	
