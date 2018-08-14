function [J, grad] = nnCostFunction(nn_params,i,h1,o,X,y,lambda)
Theta1 = reshape(nn_params(1:h1*(i + 1)),h1, (i + 1));
Theta2 = reshape(nn_params((1 + (h1 * (i + 1))):end),o, (h1 + 1));

m = size(X, 1);

% Feed forward portion
X = [ones(size(X,1),1), X];
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1), a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
K = o;    
totalm = 0;

for i = 1:m
    
    a3loop = a3(i,:);
    totalK = 0;
    
    for j = 1:K;
        % setting y = 1 when the outputs match
        y_out = 0;
        if (y(i) == j)
            y_out = 1;
        end
        error = ((-1*y_out)*log(a3loop(j))) - (1-(y_out))*log(1-a3loop(j));
        totalK = totalK + error;
    end
    
    totalm = totalm + totalK;
end

J = (totalm/m);
      
% -------------------------------------------------------------

% Next, the regularization terms

Theta1Sum = Theta1(:,2:end).*Theta1(:,2:end);
[m1,n1] = size(Theta1Sum);
Theta1Sum = ones(1,n1)*Theta1Sum';
Theta1Sum = (ones(1,m1))*Theta1Sum';
 
Theta2Sum = Theta2(:,2:end).*Theta2(:,2:end);
[m2,n2] = size(Theta2Sum);
Theta2Sum = ones(1,n2)*Theta2Sum';
Theta2Sum = (ones(1,m2))*Theta2Sum';


% J = J_nonreg + (lambda/(2*m))*(Theta1Sum + Theta2Sum); 
      
% -------------------------------------------------------------
% Error associated with layer 3's outputs... 192 trainers feeding thru 3 
% nodes
d3 = zeros(m,o);
for i = 1:m
    for j = 1:K;
        y_out = 0;
        if (y(i) == j)
            y_out = 1;
        end
        d3(i,j) = (a3(i,j) - y_out);
    end
end

% error associated with layer 2's outputs... 192 trainers feeding thru 2 
% nodes...
d2 = (d3*(Theta2(:,2:end)).*sigmoidGradient(z2));

% Now the accumulated gradients must be computed

Delta2 = zeros(size(Theta2));
for i = 1:m
    error = d3(i,:)'*a2(i,:);
    Delta2 = Delta2 + error;
end

Delta1 = zeros(size(Theta1));

for i = 1:m
    error = d2(i,:)'*X(i,:);
    Delta1 = Delta1 + error;
end

% Finally, the gradients:
Theta1_grad = (1/m).*(Delta1);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m).*(Theta1(:,2:end));

Theta2_grad = (1/m).*(Delta2); 
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m).*(Theta2(:,2:end));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
