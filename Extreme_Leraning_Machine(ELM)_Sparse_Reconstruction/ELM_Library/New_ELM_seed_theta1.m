function [TrainingTime,TrainingAccuracy,InputWeight,OutputWeight,H,Y] = New_ELM_seed_theta1(TrainingData_File,Elm_Type, NumberofHiddenNeurons, ActivationFunction,Feature,seed,lam)

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;
F = Feature ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_data = TrainingData_File ;
T=train_data(:,1:F)';
P=train_data(:,F+1:size(train_data,2))';
clear train_data;                                 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NumberofTrainingData=size(P,2); %Number of Snapshots
NumberofInputNeurons=size(P,1); % Number of data in one snapshot

%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
rng(seed)
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
rng(seed)
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        Hv = tempH(:);
        Hstd = std(Hv);
        Hnorm = tempH/Hstd ;
        H = radbas(Hnorm);
        %%%%%%%%%%%%%%%%%%%%%
        %H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [U,S,V] = svd(H,'econ');
% I = eye(NumberofHiddenNeurons) ;
% S = S + (lam*I) ;
% Hreg = U*S*V' ;  %% Change the output as Hreg instead of H
% inv_Hreg = V*(inv(S))*U' ;
% OutputWeight = T*inv_Hreg ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ('/home/cfsr123/Abdullah/ChenLuLibrary')

[Inv_H] = inv_svd_cut(H,lam) ;
OutputWeight = T*Inv_H ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
%OutputWeight=T * pinv(H);                        % implementation without regularization factor //refer to 2006 Neurocomputing paper


end_time_train=cputime;
TrainingTime=end_time_train-start_time_train  ;      %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=OutputWeight * H;  % Change accordingly H or Hreg                           %   Y: the actual output of the training data
%Y=OutputWeight * Hreg; 
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y))    ;           %   Calculate training accuracy (RMSE) for regression case
end
