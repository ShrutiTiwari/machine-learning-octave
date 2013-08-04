function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
	% Performs gradient descent to learn theta.
	% updates theta by taking num_iters gradient steps with learning rate alpha
	
	% Initialize some useful values
	m = length(y); % number of training examples
	J_history = zeros(num_iters, 1);
	
	for iter = 1:num_iters
	%Perform a single gradient step on the parameter vector theta. 
	%While debugging, it can be useful to print out the values of the cost function (computeCost) and gradient here.
		d = y - (X * theta) ;
		adjust = (d' * X )';
		theta = theta + (alpha / m )  * adjust;
		% Save the cost J in every iteration    
		J_history(iter) = computeCost(X, y, theta);
	end
end
