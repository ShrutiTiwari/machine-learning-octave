function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
	C = 1;
	sigma = 0.3;
	
	% This function returns the optimal C and sigma learning parameters found using the cross validation set.
	% Uses svmPredict to predict the labels on the cross validation set. For example, 
	%  predictions = svmPredict(model, Xval); will return the predictions on the cross validation set.
	%
	%  prediction error = mean(double(predictions ~= yval))
	%
	
	optimalC=0.01;
	optimalS=0.01;
	model= svmTrain(X, y, optimalC, @(x1, x2) gaussianKernel(x1, x2, optimalS)); 
	predictions=svmPredict(model,Xval);
	minE=mean(double(predictions ~= yval));
	
	%range=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
	range=[ 0.1, 0.3,1];
	
	i=0;
	for eachC = range,
		for eachS = range,
			i=i+1;
			fprintf('Executing %d of 64\n', i);
	
			model= svmTrain(X, y, eachC, @(x1, x2) gaussianKernel(x1, x2, eachS)); 
			predictions=svmPredict(model,Xval);
			error=mean(double(predictions ~= yval));
			if error< minE, 
				minE=error;
				optimalC=eachC;
				optimalS=eachS;			
			end;
			fprintf('Winner-so-far [C,sigma]=[%f, %f] with error [%f] \n', optimalC, optimalS, minE);
		end;
	end;
	
	C=optimalC;
	sigma=optimalS;
	fprintf('Optimal [C,Signma] are [%f, %f] of \n', C, sigma);

end
