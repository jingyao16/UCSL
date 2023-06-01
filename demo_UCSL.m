clc;
clear;
close all;

addpath(genpath('datasets')); 
addpath(genpath('utils')); 

load data_HS_LR.mat;
load data_MS_HR.mat;
TrainImage = double(imread('2013_IEEE_GRSS_DF_Contest_Samples_TR.tif'));
TestImage = double(imread('2013_IEEE_GRSS_DF_Contest_Samples_VA.tif'));

[h, w, z] = size(data_HS_LR);

HSI2d = hyperConvert2d(data_HS_LR);
MSI2d = hyperConvert2d(data_MS_HR);

TR2d = hyperConvert2d(TrainImage);
TE2d = hyperConvert2d(TestImage);

TrainLabel = TR2d(:, TR2d > 0);
TestLabel = TE2d(:, TE2d > 0);

% normalization
for i = 1 : size(HSI2d, 1)
    HSI2d(i, :) = mat2gray(HSI2d(i, :));
end
for i = 1 : size(MSI2d, 1)
    MSI2d(i, :) = mat2gray(MSI2d(i, :));
end

traindata_SP_hsi = HSI2d(:, TR2d > 0);
testdata_SP_hsi = HSI2d(:, TE2d > 0);

traindata_SP_msi = MSI2d(:, TR2d > 0);
testdata_SP_msi = MSI2d(:, TE2d > 0);

SP_0 = [traindata_SP_hsi; traindata_SP_msi];
SP_1 = [traindata_SP_hsi; zeros(size(traindata_SP_msi))];
SP_2 = [zeros(size(traindata_SP_hsi)); traindata_SP_msi];

SP_3 = [testdata_SP_hsi; zeros(size(testdata_SP_msi))];
SP_4 = [zeros(size(testdata_SP_hsi)); testdata_SP_msi];
SP_5 = [zeros(size(HSI2d)); MSI2d];
clear HSI2d MSI2d data_HS_LR data_MS_HR

d = 20; k = 30; 

X_tilde = [SP_0, SP_1, SP_2];

%% select/comment part 1 or 2 below
%% part 1
% % scfl
% alpha = [1e-3];
% beta = [1e0];
% gamma = [1e1];
% sigma = [1e2];
% 
% W_SP0 = full(creatLap(SP_0, k, sigma));
% W_SP1 = full(creatLap(SP_1, k, sigma));
% W_SP2 = full(creatLap(SP_2, k, sigma));
% % supervised graph construction
% dis = pdist([TrainLabel,TrainLabel,TrainLabel]');
% dis = squareform(dis);
% dis(dis>0) = -1;
% dis(dis==0) = 1;
% dis(dis<0) = 0;
% dis = dis./sum(dis,2);
% dis(1:size(SP_0,2),1:size(SP_0,2))=dis(1:size(SP_0,2),1:size(SP_0,2)).*W_SP0;
% dis(size(SP_0,2)+1:size(SP_0,2)+size(SP_1,2),size(SP_0,2)+1:size(SP_0,2)+size(SP_1,2))=dis(size(SP_0,2)+1:size(SP_0,2)+size(SP_1,2),size(SP_0,2)+1:size(SP_0,2)+size(SP_1,2)).*W_SP1;
% dis(size(SP_0,2)+size(SP_1,2)+1:size(SP_0,2)+size(SP_1,2)+size(SP_2,2),size(SP_0,2)+size(SP_1,2)+1:size(SP_0,2)+size(SP_1,2)+size(SP_2,2))=dis(size(SP_0,2)+size(SP_1,2)+1:size(SP_0,2)+size(SP_1,2)+size(SP_2,2),size(SP_0,2)+size(SP_1,2)+1:size(SP_0,2)+size(SP_1,2)+size(SP_2,2)).*W_SP2;

%% part 2

% ucfl
alpha = [1e-3];
beta = [1e-2];
gamma = [1e0];
sigma = [1e0];

W_SP0 = full(creatLap(SP_0, k, sigma));
W_SP1 = full(creatLap(SP_1, k, sigma));
W_SP2 = full(creatLap(SP_2, k, sigma));

% unsupervised graph construction
dis = repmat(W_SP0,3,3);
dis(size(SP_0,2)+1:size(SP_0,2)+size(SP_1,2),size(SP_0,2)+1:size(SP_0,2)+size(SP_1,2)) = W_SP1;
dis(size(SP_0,2)+size(SP_1,2)+1:size(SP_0,2)+size(SP_1,2)+size(SP_2,2),size(SP_0,2)+size(SP_1,2)+1:size(SP_0,2)+size(SP_1,2)+size(SP_2,2)) = W_SP2;

%% 

A = dis;
clear dis

D = sum(A, 2);
L = diag(D) - A;

Dhalf = diag(1./sqrt(D));
L = Dhalf*(L*Dhalf);

[Z, U_tilde]=UCSL_l2(X_tilde, L, d, alpha, beta, gamma);

%% Generate features via projections
%% Crossmodal learning
U_tilde = U_tilde(z+1:end,:);
f=U_tilde'*[traindata_SP_msi, testdata_SP_msi];
%% Feature normalization before feeding into classifier
for l=1:size(f,1)
    f(l,:)=double(mat2gray(f(l,:)));
end

traindata=f(:,1:length(TrainLabel));
testdata=f(:,length(TrainLabel)+1:end);

%% NN classifier with Euclidean distance 
mdl = ClassificationKNN.fit(traindata',TrainLabel','NumNeighbors',1,'distance','euclidean');

characterClass = predict(mdl,testdata'); 
[~,oa_NN,pa_NN,~,kappa_NN] = confusionMatrix_my( TestLabel, characterClass ); %%evaluation on test sample
aa_NN = mean(pa_NN);