function [ X1_NO , X2_NO , X3_NO ] = remove_outliers( X1, X2, X3 )

% Set cutoff value:
cut = 1;

% means:
avg1 = mean(X1);
avg2 = mean(X2);
avg3 = mean(X3);

% standard deviations:
sd1 = std(X1);
sd2 = std(X2);
sd3 = std(X3);

for i = 1:size(X1, 1)
    z = ((X1(i) - avg1)/sd1);
    if z < 0
        z = -z;
    end
    
    if z < cut
        if exist('X1_NO', 'var') == 0
            X1_NO = X1(i);
        else 
             X1_NO = [X1(i),X1_NO];
        end
    end
end

for i = 1:size(X2, 1)
    z = ((X2(i) - avg2)/sd2);
    if z < 0
        z = -z;
    end
    
    if z < cut
        if exist('X2_NO', 'var') == 0
            X2_NO = X2(i);
        else 
             X2_NO = [X2(i),X2_NO];
        end
    end
end
    
for i = 1:size(X3, 1)
    z = ((X3(i) - avg3)/sd3);
    if z < 0
        z = -z;
    end
    if z < cut
       if exist('X3_NO', 'var') == 0
            X3_NO = X3(i);
        else 
             X3_NO = [X3(i),X3_NO];
        end
    end
end
    
X1_NO = X1_NO'; X2_NO = X2_NO'; X3_NO = X3_NO';

end

