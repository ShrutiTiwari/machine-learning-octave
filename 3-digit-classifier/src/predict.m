function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

% Make predictions using learned neural network. Set p to a vector containing labels between 1 to num_labels.
% The max function can return the index of the max element, for more information see 'help max'. 
% If your examples are in rows, then, you can use max(A, [], 2) to obtain the max for each row.
%
enhancedInput = [ones(m,1) X];
Layer1= sigmoid(enhancedInput* Theta1' );
enhancedLayer1= [ones(size(Layer1,1),1) Layer1];
Result= sigmoid(enhancedLayer1 * Theta2' );
[prob, p]=max( Result, [], 2);
end
