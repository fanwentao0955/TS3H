clear 
load('flickr.mat');
addpath('lib');
addpath(genpath('lib/manopt'));
%% parameter set
param.lambda = 1e-4; param.theta = 1e1; %Stage1
param.alpha = 1e-5; param.beta =  1e-2; %Stage2
param.sigma = 1e-2; param.iter  = 6;
nbitset = [16,32,64,128];
eva_info = cell(1,length(nbitset));
%% centralization
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));
%% kernelization
%K_I 500 K_T 1000
[XKTrain,XKTest] = Kernelize(XTrain, XTest, 500); [YKTrain,YKTest]=Kernelize(YTrain,YTest, 1000); 
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));
%% construct pseudo-label
n = size(LTrain,1);
n_unlabel=floor(0.2 * n);
PL=LTrain;
PL((n - n_unlabel + 1) : n,:)=0;
%% TS3H
for kk= 1:length(nbitset)

param.nbits = nbitset(kk);

[HxTrain, HyTrain, HxTest, HyTest] = TS3H(XKTrain, YKTrain, PL, param, XKTest, YKTest, n_unlabel);

DHamm = pdist2(HxTest, HyTrain,'hamming');
[~, orderH] = sort(DHamm, 2);
eva_info_.Image_to_Text_MAP = mAP(orderH', LTrain, LTest);
 
DHamm = pdist2(HyTest, HxTrain,'hamming');
[~, orderH] = sort(DHamm, 2);
eva_info_.Text_to_Image_MAP = mAP(orderH', LTrain, LTest);

eva_info{1,kk} = eva_info_;
Image_to_Text_MAP = eva_info_.Image_to_Text_MAP;
Text_to_Image_MAP = eva_info_.Text_to_Image_MAP;  

fprintf('TS3H %d bits -- Image_to_Text_MAP: %.4f ; Text_to_Image_MAP: %.4f ; \n',nbitset(kk),Image_to_Text_MAP,Text_to_Image_MAP);
end