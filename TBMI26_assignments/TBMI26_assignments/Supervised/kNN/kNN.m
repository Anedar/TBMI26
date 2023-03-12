function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);
no_of_points = size(X,2);

%Creates two matrices, the distance to all training points and the smallest
%k distances.
distance = pdist2(transpose(Xt), transpose(X), 'euclidean');
min_distance = pdist2(transpose(Xt), transpose(X), 'euclidean', 'smallest', k);

for i=1:no_of_points
    min_points = min_distance(1:k,i); %For each point, take out its smallest distances.
    class_of_points=zeros(1,k);
    for j=1:k
        min_index = find(distance(:,i) == min_points(j)); %Find these distances for each point in the great distance matrix.
        class_of_points(j) = Lt(min_index); %Determine the class of the point by finding it in the Lt-vector.
    end
    labelsOut(i) = mode(class_of_points(j)); %Majority rule.
end
end