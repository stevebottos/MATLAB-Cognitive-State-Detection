function [p,a2,a3] = predict(Theta1, Theta2, X)
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);

% Computing the outputs of each layer
X = [ones(size(X,1),1), X];
z2 = X*Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2,1), 1), a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

p = zeros(m,1);
possibilities = [1,2,3];
for i = 1:size(p,1)
    [num,pos] = max(a3(i,:),[],2);
    p(i) = possibilities(pos);
end


end
