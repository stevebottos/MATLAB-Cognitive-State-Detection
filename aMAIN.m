
%% Clear cache, load data 
clear all; clc; close all
load data
format short;
pupil_diameter = data.pupil_diameter;
difficulty_label= data.difficulty_label;
time = data.time;

%% Load specific candidate data 
results = [zeros(40,1)];
% For an iteration over each candidate, replace "iteration = 1" with "1:40"
for iteration = 1
id = iteration; % possible values 1 to 40
    pd = pupil_diameter{id};
    dl= difficulty_label{id};
    t = time{id};
    fprintf('Current candidate id: %d\n\n', id)  
    
%% Give info on candidate's data set 
fprintf('Number of data points: %d\n', size(pd,1))
% Counting the number of output training examples and their positions in
% the vector. 
y_label_1 = 0; pos1 = 0;
y_label_2 = 0; pos2 = 0;
y_label_3 = 0; pos3 = 0;

for i = 1:size(dl,1)
    % for y = 1 
    if dl(i) == 1;
        y_label_1 = y_label_1 + 1;
        if y_label_1 == 1;
            pos1 = i;
        end
    end
    % for y = 2 
    if dl(i) == 2;
        y_label_2 = y_label_2 + 1;
        if y_label_2 == 1;
            pos2 = i;
        end
    end
    % for y = 3 
    if dl(i) == 3;
        y_label_3 = y_label_3 + 1;
        if y_label_3 == 1;
            pos3 = i;
        end
    end
end

fprintf('There are %d y=1 outputs. The first y=1 output occurs at %d\n', ...
    y_label_1, pos1)
fprintf('There are %d y=2 outputs. The first y=2 output occurs at %d\n', ...
    y_label_2, pos2)  
fprintf('There are %d y=3 outputs. The first y=3 output occurs at %d\n', ...
    y_label_3, pos3)
    
%% Turn the outputs and inputs into X and y matrices 
X1 = pd(pos1:(pos1+y_label_1-1)); 
y1 = dl(pos1:(pos1+y_label_1-1)); 
X2 = pd(pos2:(pos2+y_label_2-1));
y2 = dl(pos2:(pos2+y_label_2-1));
X3 = pd(pos3:(pos3+y_label_3-1));
y3 = dl(pos3:(pos3+y_label_3-1));

X= [X1;X2;X3];
y = [y1;y2;y3];
Xy = [X,y];
%% (PLOT) Visualizing some possible features 
% Set run = 1 in order to generate the plot
run = 0;
if run == 1
    x_axis = [1:1:size(y)];

    % Just the raw data
    figure(1)
    plot(x_axis,X)
    title('Raw data')

    X_Squared = X.^2;
    figure(2)
    plot(x_axis,X_Squared)
    title('X squared')
    % Taking a look at the means:
    avg1 = mean(X1)
    avg2 = mean(X2)
    avg3 = mean(X3)
    % Taking a look at the standard deviations:
    sd1 = std(X1)
    sd2 = std(X2)
    sd3 = std(X3)
end
%% ** REMARK 1 
 % I may be good to throw out the outliers. It's unlikely that a rapid
 % change in cognitive difficulty occured to cause spikes, and even if 
 % this was the case we are only concerned with a sustained cognitive
 % state, so a tighter range might be good... 
 
%% (PLOT) Remove outliers 
[ X1_NO , X2_NO , X3_NO ] = remove_outliers( X1, X2, X3 );
X_NO = [X1_NO;X2_NO;X3_NO];
x_axis = [1:1:size(X_NO)];

% without outliers
run = 1;
if run == 1
    figure(1)
    plot(x_axis,X_NO)
    title('No Outliers')
    avg1_NO = mean(X1_NO);
    avg2_NO = mean(X2_NO);
    avg3_NO = mean(X3_NO);
    % Taking a look at the standard deviations:
    sd1_NO = std(X1_NO);
    sd2_NO = std(X2_NO);
    sd3_NO = std(X3_NO);
end

%% ** REMARK 2
    % One thing that sticks out to my quite a bit with the outliers thrown
    % away are the distinct upper and lower bounds of each difficulty level... 
    % Let's see what those are
    
%% (TABLE) Determining upper and lower bounds of the data set

run = 1;
if run == 0
    % Set s to some point in the dataset that you'd like to observe.
    % This comes into play later in this run sequence, since I am
    # interested in comparing the upper and lower bounds of the 
    # whole dataset to the upper and lower bounds of some small
    # slice of the dataset. 
    s = 3428; % sample size start
    ss = s+150; % sample size finish
    
    fprintf('Over all samples');
    fprintf('\n\t\tMax\t\t\t\tMin\t\t\t\tMedian\n')
    mid1 = ((max(X1_NO) + min(X1_NO))/2)*1000;
    mid2= ((max(X2_NO) + min(X2_NO))/2)*1000;
    mid3 = ((max(X3_NO) + min(X3_NO))/2)*1000;
    fprintf('X1\t%5d\t%5d\t%5d\n', max(X1_NO),min(X1_NO),mid1) 
    fprintf('X2\t%5d\t%5d\t%5d\n', max(X2_NO), min(X2_NO),mid2)
    fprintf('X3\t%5d\t%5d\t%5d\n', max(X3_NO), min(X3_NO),mid3)
    
    fprintf('\nOver 150 samples:')
    fprintf('\n\t\tMax\t\t\t\tMin\t\t\t\tMedian\n')
    mid1b = ((max(X1_NO(s:ss))) + min(X1_NO(s:ss)))/2*1000;
    mid2b= ((max(X2_NO(s:ss))) + min(X2_NO(s:ss)))/2*1000;
    mid3b = ((max(X3_NO(s:ss))) + min(X3_NO(s:ss)))/2*1000;
    fprintf('X1\t%5d\t%5d\t%5d\n', max(X1_NO(s:ss)),min(X1_NO(s:ss)),mid1b) 
    fprintf('X2\t%5d\t%5d\t%5d\n', max(X2_NO(s:ss)), min(X2_NO(s:ss)),mid2b)
    fprintf('X3\t%5d\t%5d\t%5d\n', max(X3_NO(s:ss)), min(X3_NO(s:ss)),mid3b)
end

%% ** REMARK 3
    % The results here look good. The upper and lower bounds of each
    % dataset (X1, X2, and X3) are typically quite distinct. This 
    % distinction persists in small slices of the data as well.

%% Splitting non-outlier sets into a training set and a test set
X1_train = X1_NO(1:floor(0.6*(size(X1_NO,1))));
X2_train = X2_NO(1:floor(0.6*(size(X2_NO,1))));
X3_train = X3_NO(1:floor(0.6*(size(X3_NO,1))));

X1_test = X1_NO(floor(0.6*(size(X1_NO,1)))+1:end);
X2_test = X2_NO(floor(0.6*(size(X2_NO,1)))+1:end);
X3_test = X3_NO(floor(0.6*(size(X3_NO,1)))+1:end);

%% (PLOT) Finally, build the features from the training set's values...

sample_size = 150;
features = 2;

% ts stands for "training set"
[ts1, ts2, ts3] = createTrainSet( sample_size, features, X1_train, ...
                                            X2_train, X3_train );
figure(2)
plot(ts1(:,1),ts1(:,2),'rd',ts2(:,1),ts2(:,2),'bd',ts3(:,1),ts3(:,2),'gd')
legend('y=1','y=2','y=3')

%% ** REMARK 4
% As can be seen from the plot, the boundaries are very distinct and a
% simple logistic regression classifier should suffice to create boundaries.
% Nonetheless, for practice I'll develop a neural network to handle this.

%% Setting up the actual training sets using the previously found features

% first build one training set, append a column of ones for the bias node
ts1 = [ts1,ones(size(ts1,1),1)];
ts2 = [ts2,ones(size(ts2,1),1).*2];
ts3 = [ts3,ones(size(ts3,1),1).*3];

trainset = [ts1;ts2;ts3];
y = trainset(:,3);
trainset = trainset(:,1:2);
trainset = trainset.*1000;
trainset = trainset.^6;
trainset = trainset./1000;

%% Setting neural network params
% i = input layer size
% h1 = first hidden layer size
% o = output layer size

i = size(trainset,2); % Since the columns in trainset represent features
h1 = 2; % not including the bias node
o = 3; % number of output layer nodes

% Randomly initializing Theta arrays
Theta1 = randInitializeWeights(i,h1);
Theta2 = randInitializeWeights(h1,o);
nn_params = [Theta1(:); Theta2(:)];
initial_nn_params = nn_params;

% No regularization for now
lambda = 0;

% Uncomment the next two lines if you'd like to save the training set and its labels
% csvwrite('trainset.csv',trainset)
% csvwrite('y.csv',y)

% Producing the cost function and the gradient. This function is sent to an optimization
% function in order to minimize the cost
[J, grad] = nnCostFunction(nn_params, i, h1, o, trainset, y, lambda);

options = optimset('MaxIter', 50);

% Create "short hand" for the cost function to be minimized. Want to minimize J by
% minimizing the Theta arrays, so point to the theta argument
costFunction = @(p) nnCostFunction(p, i, h1, o, trainset, y, lambda );

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters). The output of interest after optmization are
% the optimized Theta values
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%% Testing accuracy on test set
Theta1 = reshape(nn_params(1:h1*(i + 1)),h1, (i + 1));
Theta2 = reshape(nn_params((1 + (h1 * (i + 1))):end),o, (h1 + 1));
[ts1, ts2, ts3] = createTrainSet( sample_size, features, X1_test, ...
                                            X2_test, X3_test );

test1 = [ts1,ones(size(ts1,1),1)];
test2 = [ts2,ones(size(ts2,1),1).*2];
test3 = [ts3,ones(size(ts3,1),1).*3];

test = [test1;test2;test3];

y_test = test(:,3);

test = test(:,1:2);
test = test.*1000;
test = test.^6;
test = test./1000;

% csvwrite('testset.csv',test)
% csvwrite('ytest.csv',y_test)

% The actual testing step to display accuracy
[p,a2,a3] = predict(Theta1, Theta2, test );
p_train = predict(Theta1, Theta2, trainset);
score = zeros(size(p));
m = size(y_test,1);
for i = 1:size(p,1)
    if p(i) == y_test(i)
        score(i) = 1;
    end
end
acc = (sum(score)/m)*100;
fprintf('Accuracy: %0.2f%c\n', acc,'%') 
results(iteration,1) = acc;
end

% Show the results of each run
results;

