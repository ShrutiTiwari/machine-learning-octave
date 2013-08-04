function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
	X = reshape(params(1:num_movies*num_features), num_movies, num_features);
	Theta = reshape(params(num_movies*num_features+1:end), ...
	                num_users, num_features);
	
	J = 0;
	X_grad = zeros(size(X));
	Theta_grad = zeros(size(Theta));
	
	% Compute the cost function and gradient for collaborative filtering. 
	% First implement the cost function (without regularization) and make sure it is matches our costs. 
	% After that, implement the gradient and use the checkCostFunction routine to check that the gradient is correct. 
	% Finally, implement regularization.
	%
	% Notes: X - num_movies  x num_features matrix of movie features
	%        Theta - num_users  x num_features matrix of user features
	%        Y - num_movies x num_users matrix of user ratings of movies
	%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
	%            i-th movie was rated by the j-th user
	% Set the following variables correctly:
	%        X_grad - num_movies x num_features matrix, containing the 
	%                 partial derivatives w.r.t. to each element of X
	%        Theta_grad - num_users x num_features matrix, containing the 
	%                     partial derivatives w.r.t. to each element of Theta
	%

	M=X*Theta'- Y;
	J=sum(sum((R .* M) .^ 2))/2;
	
	for i = 1:num_movies,
		idx=find(R(i,:)==1);
		Thetat=Theta(idx,:);
		Yt=Y(i,idx);
		X_grad(i,:) =(X(i,:)* Thetat' - Yt)*Thetat;
	end;
	
	for j = 1:num_users,
		idx=find(R(:,j)==1);
		Xt=X(:,idx);
		Yt=Y(idx,j);
		Theta_grad(:,j) =(Xt* Theta(:,j)' - Yt)*Theta;
	end;
	
	grad = [X_grad(:); Theta_grad(:)];

end
