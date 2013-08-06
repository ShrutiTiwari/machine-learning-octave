function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
	m = size(X, 1);
	
	error_train = zeros(m, 1);
	error_val   = zeros(m, 1);
	
	% This function returns training errors in error_train and the cross validation errors in error_val. 
	% i.e., error_train(i) and error_val(i) should give the errors obtained after training on i examples.
	
	% Training error is evaluated on the first i training examples (i.e., X(1:i, :) and y(1:i)).
	% For the cross-validation error, evaluate on the _entire_ cross validation set (Xval and yval).
	%
	% Using cost function (linearRegCostFunction) to compute the training and cross validation error, 
	% with the lambda argument set to 0. 
	% lambda is used when running the training to obtain the theta parameters.
	%
	% Loop over the examples with the following:
	%       for i = 1:m
	%           % Compute train/cross validation errors using training examples 
	%           % X(1:i, :) and y(1:i), storing the result in 
	%           % error_train(i) and error_val(i)
	%           ....
	numVal=size(Xval, 1);
	for(i=1:m),	
		trainX=X(1:i, :);
		trainy=y(1:i);
		theta = trainLinearReg(trainX, trainy,lambda);
	
		error_train(i)=sum((trainX*theta - trainy) .^ 2)/(2*i);
		error_val(i)=sum((Xval*theta - yval).^2)/(2*numVal);
	end;
end
