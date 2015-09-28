function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    new_theta = theta;

    for j = 1:length(theta)
        change = 0;
        for counter = 1:m
            %Theta is also, obviously, a vector, although octave starts at 1 indexing
            % so theta[0] is theta(1) and theta[1] is theta(2)
            hyp = 0;
            for inner_counter = 1:length(theta)
                hyp += theta(inner_counter) * X(counter, inner_counter);
            endfor
            %fprintf('theta(1) = %f theta(2) = %f X(%f) = %f, theta(1) + theta(2) * X(%f) = %f\n', theta(1), theta(2), counter, X(counter),counter, hyp);
            change += (hyp - y(counter)) * X(counter, j);
        endfor

        change *= (alpha/m);

        new_theta(j) -= change; %Adjust only the new vector, so we get simultaneous update
    endfor

    theta = new_theta;

    cost = computeCost(X, y, theta);

    if (cost==0)
        break;
    endif

    % Save the cost J in every iteration    
    J_history(iter) = cost;

end

end
