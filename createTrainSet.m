function [ ts1, ts2, ts3 ] = createTrainSet( sample_size, features, X1_train, ...
                                            X2_train, X3_train )

% Some shorthands
s = sample_size;
f = features;
X1 = X1_train;
X2 = X2_train;
X3 = X3_train;

% putting X1-X3 into matrices of the sample size (cols)
stop1 = floor(size(X1,1)/s);
X1 = reshape(X1(1:stop1*s),stop1,s);

stop2 = floor(size(X2,1)/s);
X2 = reshape(X2(1:stop2*s),stop2,s);

stop3 = floor(size(X3,1)/s);
X3 = reshape(X3(1:stop3*s),stop3,s);

% finally the training sets
ts1 = [max(X1,[],2), min(X1,[],2)];
ts2 = [max(X2,[],2), min(X2,[],2)];
ts3 = [max(X3,[],2), min(X3,[],2)];
% take the min and max in each row and use them as your two features
end

