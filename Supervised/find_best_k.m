%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 3; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

plotCase(X,D);

%% Select a subset of the training features

numCrossBins = 3; % Number of Bins you want to devide your data into
numSamplesPerLabelPerCrossBin = 50; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerCrossBin, numCrossBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};

%%Find best kNN through cross validation
bestK = -1;
bestAcc = 0;

for k = 1:9
    systemAcc = 0;
    %Cross validation
    for i = 1:numCrossBins
        xTest = Xt{i};
        %Setting the training data, there are special cases that needs to 
        %be taken into consideration. For example, if numCrossBins = 3 and
        %i = 1, the training data must be parts 3 and 2 of the data set.
        if i == 1
            h = numCrossBins;
        else
            h = i-1; 
        end
        if i == numCrossBins
            j = 1;
        else
            j = i+1;
        end
        xTrain = [Xt{h}, Xt{j}];
        
        %Set the labels in the same fashions as the features.
        lTest = Lt{i};
        lTrain = [Lt{h}, Lt{j}];
        
        %Run the kNN algorithm for the training and test data set above.
        LkNN = kNN(xTest, k, xTrain, lTrain);
        cM = calcConfusionMatrix(LkNN, lTest);
        acc = calcAccuracy(cM);
        systemAcc = systemAcc + acc;
    end
    %Calculate mean accuracy.
    systemAcc = systemAcc/numCrossBins;
    if systemAcc > bestAcc;
        bestK = k;
        bestAcc = systemAcc;
    end
end

numTestBins = 2; % Number of Bins you want to devide your data into
numSamplesPerLabelPerTestBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerTestBin, numTestBins, selectAtRandom );

LkNN = kNN(Xt{2}, bestK, Xt{1}, Lt{1});

cM = calcConfusionMatrix(LkNN, Lt{2});

bestK
acc = calcAccuracy(cM)

%% Plot classifications
% Note: You do not need to change this code.
if dataSetNr < 4
    plotkNNResultDots(Xt{2},LkNN,k,Lt{2},Xt{1},Lt{1});
else
    plotResultsOCR( Xt{2}, Lt{2}, LkNN )
end