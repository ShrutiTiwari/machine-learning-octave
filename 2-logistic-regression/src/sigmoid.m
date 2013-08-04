function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
	
	g = zeros(size(z));
	% Sigmoid of each value of z (z can be a matrix,vector or scalar).
	g=1 ./ (1.+ exp((-1)*z));
end
