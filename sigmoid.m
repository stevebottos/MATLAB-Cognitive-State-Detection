function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
% g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% We are given an input z, which will be X*theta in this case...

g = 1./(1+exp(-1.*z)); % ** USE exp(), NOT e()!!

% =============================================================

end
