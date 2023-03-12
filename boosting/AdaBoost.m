% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces);
nonfaces = double(nonfaces);

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

% Generate Haar feature masks
nbrHaarFeatures = 50;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 1500;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
% Creates our "limits" by creating a temporary sorted version of the 
% training data and placing the borders inbetween each step as well as
% in the beginning and end as well.
t = zeros(size(xTrain,1),size(xTrain,2)+1);
tempX = sort(xTrain,2);
for j = 1:size(xTrain,1)
    for i=2:size(t,2)-1
        t(j,i) = tempX(j,i-1) + (tempX(j,i)-tempX(j,i-1))/2;
    end
    t(j,1) = tempX(j,1)-(tempX(j,2) - tempX(j,1))/2;
    t(j,end) = tempX(j,end)+(tempX(j,end)-tempX(j,end-1))/2;
end

nbrWeakClassifiers = 100;
d = zeros(nbrWeakClassifiers+1,size(xTrain,2));
d(1,:) = 1/size(xTrain,2)*ones(1,size(xTrain,2)); %startvikter
alpha = zeros(4,nbrWeakClassifiers);

%Fungerar, men fungerar den bra? 
for k = 1:nbrWeakClassifiers
    eps = 0; %startepsilon
    epsmin = [2, 0, 0, 0];
    Cmin = 2*ones(1,size(xTrain,2));
    for i = 1:size(xTrain,1)
        P=1;
        for j = 1:size(t,2)
            C = WeakClassifier(t(i,j),P,xTrain(i,:));
            eps = WeakClassifierError(C,d(k,:),yTrain);
            if 0.5 < eps && eps < 1
                eps = 1 - eps;
                P = -P;
                C = -C;
            end
            if eps < epsmin(1)
                epsmin = [eps, i, j, P];
                Cmin = C;
                
            end
        end
    end
    Cmin;
    %trainepserr(i) = 1-epsmin(1);
    alpha(:,k) = [1/2*log((1-epsmin(1))/epsmin(1)); epsmin(2); epsmin(3); epsmin(4)] ;
    Cmin = 2*(Cmin~=yTrain)-1;
    d(k+1,:) = d(k,:).*exp(alpha(1,k)*Cmin);
    d(k+1,:) = 1/sum(d(k+1,:))*d(k+1,:);
end
%% Extract test data

nbrTestExamples = 2500;

testImages  = cat(3,faces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)),...
                    nonfaces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

%% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.


res = zeros(1,size(xTest,2));
res2 = zeros(1,size(xTrain,2));
testepserr = zeros(1,nbrWeakClassifiers);
trainepserr = zeros(1,nbrWeakClassifiers);
for i=1:nbrWeakClassifiers
    res = res + alpha(1,i)*WeakClassifier(t(alpha(2,i), alpha(3,i)), alpha(4,i), xTest(alpha(2,i),:));
    res2 = res2 + alpha(1,i)*WeakClassifier(t(alpha(2,i), alpha(3,i)), alpha(4,i), xTrain(alpha(2,i),:));
    testepserr(i) = sum(sign(res)==yTest)/length(yTest);
    trainepserr(i) = sum(sign(res2)==yTrain)/length(yTrain);
end
sum(sign(res)==yTest)/length(yTest)
sum(sign(res2)==yTrain)/length(yTrain)
%% Plot the error of the strong classifier as  function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.
figure(4);
plot(1:nbrWeakClassifiers, testepserr, 'b', 1:nbrWeakClassifiers, trainepserr, 'r')
legend('Test data', 'Training data')