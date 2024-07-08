function [HxTrain,HyTrain,HxTest,HyTest] = TS3H(X, Y, L, param, XTest, YTest, n_unlabel)

n = size(X,1);
Xu=X((n - n_unlabel + 1) : n,:); Xl=X(1 : (n - n_unlabel),:);
Yu=Y((n - n_unlabel + 1) : n,:); Yl=Y(1 : (n - n_unlabel),:);
X = X'; Y=Y'; L=L'; 
Xl = Xl'; Yl = Yl'; 
Xu = Xu'; Yu = Yu'; 
c = size(L,1);

nbits = param.nbits;
alpha = param.alpha;
beta = param.beta;
lambda = param.lambda;
theta = param.theta;
sigma = param.sigma;

% initialize Hash codes 
sel_sample = Y(:,randsample(n, 2000),:);
[pcaW, ~] = eigs(cov(sel_sample'), nbits);
E = pcaW'*Y;
H = sign(E);
H(H==0) = -1;

%Stage1
Ll = L(:,1 : (n - n_unlabel));
U = (NormalizeFea(Ll',1))';
Ll_=(NormalizeFea(Ll',1))';

for iter = 1:param.iter   
    
    WA1 = (eye(c)+ theta * U * U')\(lambda * eye(c));
    WB1 = Xl * Xl';
    WC1 = (eye(c)+ theta * U * U')\(U * Xl' + theta * U * Ll_' * Ll_ * Xl');
    W1 = sylvester((WA1),(WB1),(WC1));

    WA2 = (eye(c)+ theta * U * U')\(lambda * eye(c));
    WB2 = Yl * Yl';
    WC2 = (eye(c)+ theta * U * U')\(U * Yl' + theta * U * Ll_' * Ll_ * Yl');
    W2 = sylvester((WA2),(WB2),(WC2));
    
    U = (2 * eye(c)  + theta * W1 * Xl * Xl' * W1' + theta * W2 * Yl * Yl' * W2')...
        \ (W1 * Xl + W2 * Yl + theta * W1 * Xl * Ll_'* Ll_ + theta * W2 * Yl * Ll_'* Ll_);
    
end

Lu = 0.5 * W1 * Xu + 0.5 * W2 * Yu;
Lu (Lu < 0) = 0;
L(:,(n - n_unlabel + 1) : n) = Lu;

%Stage2
L_ = (NormalizeFea(L',1))';

for iter = 1:param.iter 
    
    H = sign(nbits*alpha*E*L_'*L_ + beta*E);
    P = (L*E')/(E*E'+ (1e-4)*eye(nbits));
    
    K = P'* L + beta*H + alpha*nbits*H*L_'*L_;
    K = K';
    Temp = K'*K-1/n*(K'*ones(n,1)*(ones(1,n)*K));
    [~,Lmd,QQ] = svd(Temp); clear Temp
    idx = (diag(Lmd)>1e-4);
    Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
    Pt = (K-1/n*ones(n,1)*(ones(1,n)*K)) *  (Q / (sqrt(Lmd(idx,idx))));
    P_ = orth(randn(n,nbits-length(find(idx==1))));
    E = sqrt(n)*[Pt P_]*[Q Q_]';
    E = E';  
end

%Stage3
H1 = H;
H2 = H;
T1= H * X' /(X * X' + sigma * eye(size(X,1)));
T2= H * Y' /(Y * Y' + sigma * eye(size(Y,1)));
HxTest = double(XTest*T1'>0);
HyTest = double(YTest*T2'>0);
HxTrain = double(H1 > 0)';
HyTrain = double(H2 > 0)';

end

