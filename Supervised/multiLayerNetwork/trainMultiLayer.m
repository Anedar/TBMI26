function [Wout,Vout, trainingError, testError ] = trainMultiLayer(Xtraining,Dtraining,Xtest,Dtest, W0, V0,numIterations, learningRate )
%TRAINMULTILAYER Trains the network (Learning)
%   Inputs:
%               X* - Trainin/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               V0 - Weights of the output neurons (matrix)
%               W0 - Weights of the output neurons (matrix)
%               numIterations - Number of learning setps (scalar)
%               learningRate - The learningrate (scalar)
%
%   Output:
%               Wout - Weights after training (matrix)
%               Vout - Weights after training (matrix)
%               trainingError - The training error for each iteration
%                               (vector)
%               testError - The test error for each iteration
%                               (vector)

% Initiate variables
trainingError = nan(numIterations+1,1);
testError = nan(numIterations+1,1);
numTraining = size(Xtraining,2);
numTest = size(Xtest,2);
numClasses = size(Dtraining,1) - 1;
Wout = W0;
Vout = V0;

% Calculate initial error
Ytraining = runMultiLayer(Xtraining, W0, V0);
Ytest = runMultiLayer(Xtest, W0, V0);
trainingError(1) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
testError(1) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);

for n = 1:numIterations
    Ytraining = runMultiLayer(Xtraining, Wout, Vout);
    
    S = Wout*Xtraining; %Calculate the sumation of the weights and the input signals (hidden neuron)
    U = tanh(S); %Calculate the activation function as a hyperbolic tangent
    T = Ytraining;

    %The gradients without an activation function on the output.
    %grad_v = 2/numTraining*(Ytraining-Dtraining)*U'; %Calculate the gradient for the output layer
    %grad_w = 2/numTraining*((Vout'*(Ytraining-Dtraining)).*(1-U.^2))*Xtraining'; %..and for the hidden layer.
    
    %With tanh on the output too...
    grad_v = (2/numTraining)*((T-Dtraining).*(1 - T.^2))*U';
    grad_w = (2/numTraining)*(Vout'*((T - Dtraining).*(1 - T.^2)).*(1 - U.^2))*Xtraining';


    Wout = Wout - learningRate * grad_w; %Take the learning step.
    Vout = Vout - learningRate * grad_v; %Take the learning step.

    Ytraining = runMultiLayer(Xtraining, Wout, Vout);
    Ytest = runMultiLayer(Xtest, Wout, Vout);

    trainingError(1+n) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
    testError(1+n) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);
end

end
