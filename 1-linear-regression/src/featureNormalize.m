function [X_norm, mu, sigma] = featureNormalize(X)
	% Normalizes the features in X - returns a normalized version of X where
	% the mean value of each feature is 0 and the standard deviation is 1. 
	% This is often a good preprocessing step to do when working with learning algorithms.

	X_norm = X;
	mu = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2));
	
	% For each feature dimension, 
	% i. compute the mean of the feature and subtract it from the dataset (store in mu) 
	% ii. compute the standard deviation of each feature and divide each feature by it's standard deviation (store in sigma) 
	% Note that X is a matrix where each column is a feature and each row is an example. 
	% You need to perform the normalization separately for each feature. 
	for iter = 1:size(X,2)
		feature=X(:,iter);
		index=iter;
		mu(1,index)= mean(feature);
		sigma(1,index)=std(feature);
		X_norm(:,iter)=(1/sigma(1,index))* (feature .- mu(1,index));
	end
	mu
	sigma
end
