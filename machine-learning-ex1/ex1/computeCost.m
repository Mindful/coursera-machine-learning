function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

%"As you are doing this, remember that the
%variables X and y are not scalar values, but matrices whose rows represent
%the examples from the training set."


% Initialize some useful values
m = length(y); % number of training examples

%Sum J 
%This would be better vectorized I assume, but for now simple and easy is fine
J = 0;
%h θ (x) = θ0 + θ1 * x
for counter = 1:m
	%Theta is also, obviously, a vector, although octave starts at 1 indexing
	% so theta[0] is theta(1) and theta[1] is theta(2)
	hyp = 0;
	for inner_counter = 1:length(theta)
		hyp += theta(inner_counter) * X(counter, inner_counter);
	endfor
	%fprintf('theta(1) = %f theta(2) = %f X(%f) = %f, theta(1) + theta(2) * X(%f) = %f\n', theta(1), theta(2), counter, X(counter),counter, hyp);
	J += (hyp - y(counter))^2;
endfor



J /= (2*m);

end
