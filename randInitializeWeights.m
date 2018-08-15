function W = randInitializeWeights(L_in, L_out)
% Returns a set of randomly initialized theta values
% L_in = size of relative input layer, not including bias node
% L_out = size of relative output layer, not including bias node

% Initialize the array
W = zeros(L_out, 1 + L_in);

ep = sqrt(6)/(sqrt(L_in+L_out));
W = rand(L_out, 1+L_in)*2*ep - ep;


end 
