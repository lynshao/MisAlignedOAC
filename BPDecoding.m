function real_sum_seq = BPDecoding(InputSeq,para_seq1,para_seq2)
%% Input
M = 4;
L = para_seq2(2*M+2);
aa = M*(L+1) - 1;
samples = InputSeq(1:aa) + 1j * InputSeq(aa+1:end);

dd = para_seq1(1:M);
hh = para_seq1((M+1):(2*M)) + 1j * para_seq1((2*M+1):(3*M));
hh = reshape(hh,M,1);
sourceMean = para_seq2(1:M);
sourceMean = reshape(sourceMean,M,1);
variance = para_seq2((M+1):(2*M));
variance = reshape(variance,M,1);
NoisePower = para_seq2(2*M+1);

%% Graphical decoding -- applies to M = 4 only
% -------------------------------------- Prepare the prior Gaussian messages (Eta,LambdaMat)
prior_Sigma1 = diag([variance(1),0,0,0,variance(1),0,0,0]);
prior_Sigma2 = diag([variance(1),variance(2),0,0,variance(1),variance(2),0,0]);
prior_Sigma3 = diag([variance(1),variance(2),variance(3),0,variance(1),variance(2),variance(3),0]);
prior_Sigma4 = diag([variance',variance']);
prior_Sigma5 = diag([0,variance(2),variance(3),variance(4),0,variance(2),variance(3),variance(4)]);
prior_Sigma6 = diag([0,0,variance(3),variance(4),0,0,variance(3),variance(4)]);
prior_Sigma7 = diag([0,0,0,variance(4),0,0,0,variance(4)]);

prior_Lamb1 = pinv(prior_Sigma1);
prior_Lamb2 = pinv(prior_Sigma2);
prior_Lamb3 = pinv(prior_Sigma3);
prior_Lamb4 = pinv(prior_Sigma4);
prior_Lamb5 = pinv(prior_Sigma5);
prior_Lamb6 = pinv(prior_Sigma6);
prior_Lamb7 = pinv(prior_Sigma7);

prior_mu1 = [sourceMean(1);0;0;0;sourceMean(1);0;0;0];
prior_mu2 = [sourceMean(1);sourceMean(2);0;0;sourceMean(1);sourceMean(2);0;0];
prior_mu3 = [sourceMean(1);sourceMean(2);sourceMean(3);0;sourceMean(1);sourceMean(2);sourceMean(3);0];
prior_mu4 = [sourceMean;sourceMean];
prior_mu5 = [0;sourceMean(2);sourceMean(3);sourceMean(4);0;sourceMean(2);sourceMean(3);sourceMean(4)];
prior_mu6 = [0;0;sourceMean(3);sourceMean(4);0;0;sourceMean(3);sourceMean(4)];
prior_mu7 = [0;0;0;sourceMean(4);0;0;0;sourceMean(4)];

prior_eta1 = prior_Lamb1 * prior_mu1;
prior_eta2 = prior_Lamb2 * prior_mu2;
prior_eta3 = prior_Lamb3 * prior_mu3;
prior_eta4 = prior_Lamb4 * prior_mu4;
prior_eta5 = prior_Lamb5 * prior_mu5;
prior_eta6 = prior_Lamb6 * prior_mu6;
prior_eta7 = prior_Lamb5 * prior_mu7;

% -------------------------------------- Prepare the Gaussian messages (Eta,LambdaMat) obtained from the observation nodes
% LambdaMat = covariance^{-1}
beta1 = [real(hh),imag(hh)];
beta2 = [-imag(hh),real(hh)];
Obser_Lamb = [beta1*beta1.',beta1*beta2.';(beta1*beta2.').',beta1*beta1.'];
element = [[1,zeros(1,3)]; zeros(1,4); zeros(1,4); zeros(1,4)];
ObserMat1 = repmat(element,2,2) .* Obser_Lamb;
element = [[ones(1,2),zeros(1,2)]; [ones(1,2),zeros(1,2)]; zeros(1,4); zeros(1,4)];
ObserMat2 = repmat(element,2,2) .* Obser_Lamb;
element = [[ones(1,3),zeros(1,1)]; [ones(1,3),zeros(1,1)]; [ones(1,3),zeros(1,1)]; zeros(1,4)];
ObserMat3 = repmat(element,2,2) .* Obser_Lamb;
element = ones(M,M);
ObserMat4 = repmat(element,2,2) .* Obser_Lamb;
element = [zeros(1,4); [zeros(1,1),ones(1,3)];[zeros(1,1),ones(1,3)];[zeros(1,1),ones(1,3)]];
ObserMat5 = repmat(element,2,2) .* Obser_Lamb;
element = [zeros(1,4);zeros(1,4);[zeros(1,2),ones(1,2)];[zeros(1,2),ones(1,2)]];
ObserMat6 = repmat(element,2,2) .* Obser_Lamb;
element = [zeros(1,4);zeros(1,4);zeros(1,4);[zeros(1,3),ones(1,1)]];
ObserMat7 = repmat(element,2,2) .* Obser_Lamb;

% Eta = LambdaMat * mean
etaMat = [beta1;beta2]*[real(samples);imag(samples)];
% process the boundaries
etaMat([2,3,4,6,7,8],1) = 0;
etaMat([3,4,7,8],2) = 0;
etaMat([4,8],3) = 0;
etaMat([1,5],end-1) = 0;
etaMat([1,2,5,6],end-1) = 0;
etaMat([1,2,3,5,6,7],end) = 0;
% noise variance
VarVec = NoisePower/2./dd;
% ------------------------------------------------------------ right message passing
R_m3_eta = zeros(2*M,M*(L+1)-1);
R_m3_Lamb = zeros(2*M,2*M,M*(L+1)-1);
for idx = 1 : M*(L+1)-2
    % ----------------------------- message m1: Gaussian
    m1_eta = etaMat(:,idx) / VarVec(mod(idx-1,M)+1);
    if idx == 1 % first boundary -- will only be used in the right passing
        ObserMat = ObserMat1;
        prior_eta = prior_eta1;
        prior_Lamb = prior_Lamb1;
    elseif idx == 2 % second boundary
        ObserMat = ObserMat2;
        prior_eta = prior_eta2;
        prior_Lamb = prior_Lamb2;
    elseif idx == 3 % third boundary
        ObserMat = ObserMat3;
        prior_eta = prior_eta3;
        prior_Lamb = prior_Lamb3;
    elseif idx == M*(L+1)-3 % second last boundary
        ObserMat = ObserMat5;
        prior_eta = prior_eta5;
        prior_Lamb = prior_Lamb5;
    elseif idx == M*(L+1)-2 % second last boundary
        ObserMat = ObserMat6;
        prior_eta = prior_eta6;
        prior_Lamb = prior_Lamb6;
%     elseif idx == M*(L+1)-1 % last boundary -- will only be used in the left passing
%         ObserMat = ObserMat7;
%         prior_eta = prior_eta7;
%         prior_Lamb = prior_Lamb7;
    else
        ObserMat = ObserMat4;
        prior_eta = prior_eta4;
        prior_Lamb = prior_Lamb4;
    end
    m1_Lamb = ObserMat / VarVec(mod(idx-1,M)+1);

    % ----------------------------- message m2: product
    if idx == 1 % first boundary
        m2_eta = m1_eta + prior_eta;
        m2_Lamb = m1_Lamb + prior_Lamb;
    else
        m2_eta = m1_eta + prior_eta + R_m3_eta(:,idx-1);
        m2_Lamb = m1_Lamb + prior_Lamb + R_m3_Lamb(:,:,idx-1);
    end

    % ----------------------------- message m3: sum
    m2_Sigma = pinv(m2_Lamb); % find the matrix Sigma of m2
    pos = [mod(idx,M)+1, mod(idx,M)+1+M]; % pos of two variables (real and imag) to be integrated
    % convert m2_eta back to m2_mean to delete columns -> convert back and add zero columns -> get the new m3_eta
    m2_mean = m2_Sigma * m2_eta; % m2_mean
    m2_mean(pos) = []; % set to zero and convert back to eta (see below)
    % convert m2_Lambda back to m2_Sigma to delete rows/columns -> convert back and add zero rows/columns -> get the new m3_Lambda
    m2_Sigma(pos,:) = []; % delete the rows and columns of m2_Sigma
    m2_Sigma(:,pos) = [];
    m3_Lamb = pinv(m2_Sigma); % the dimension-reduced m3_lambda
    m3_Lamb_add0 = [m3_Lamb(1:(pos(1)-1),:); zeros(1,2*M-2); m3_Lamb(pos(1):pos(1)+M-2,:); zeros(1,2*M-2); m3_Lamb(pos(1)+M-1:end,:)]; % insert two all-zero rows to m3_lambda
    m3_Lamb_add0 = [m3_Lamb_add0(:,1:(pos(1)-1)), zeros(2*M,1), m3_Lamb_add0(:,pos(1):pos(1)+M-2), zeros(2*M,1), m3_Lamb_add0(:,pos(1)+M-1:end)]; % insert two all-zero columns to m3_lambda
    m3_eta = m3_Lamb * m2_mean;
    m3_eta_add0 = [m3_eta(1:(pos(1)-1)); 0; m3_eta(pos(1):pos(1)+M-2); 0; m3_eta(pos(1)+M-1:end)];
    % ----------------------------- store m3
    R_m3_eta(:,idx) = m3_eta_add0;
    R_m3_Lamb(:,:,idx) = m3_Lamb_add0;
end

% ------------------------------------------------------------ left message passing
L_m3_eta = zeros(2*M,M*(L+1)-1);
L_m3_Lamb = zeros(2*M,2*M,M*(L+1)-1);
for idx = M*(L+1)-1 : -1 : 2
    % ----------------------------- message m1: Gaussian
    m1_eta = etaMat(:,idx) / VarVec(mod(idx-1,M)+1);
    if idx == 1 % first boundary -- will only be used in the right passing
        ObserMat = ObserMat1;
        prior_eta = prior_eta1;
        prior_Lamb = prior_Lamb1;
    elseif idx == 2 % second boundary
        ObserMat = ObserMat2;
        prior_eta = prior_eta2;
        prior_Lamb = prior_Lamb2;
    elseif idx == 3 % third boundary
        ObserMat = ObserMat3;
        prior_eta = prior_eta3;
        prior_Lamb = prior_Lamb3;
    elseif idx == M*(L+1)-3 % second last boundary
        ObserMat = ObserMat5;
        prior_eta = prior_eta5;
        prior_Lamb = prior_Lamb5;
    elseif idx == M*(L+1)-2 % second last boundary
        ObserMat = ObserMat6;
        prior_eta = prior_eta6;
        prior_Lamb = prior_Lamb6;
    elseif idx == M*(L+1)-1 % last boundary -- will only be used in the left passing
        ObserMat = ObserMat7;
        prior_eta = prior_eta7;
        prior_Lamb = prior_Lamb7;
    else
        ObserMat = ObserMat4;
        prior_eta = prior_eta4;
        prior_Lamb = prior_Lamb4;
    end
    m1_Lamb = ObserMat / VarVec(mod(idx-1,M)+1);

    % ----------------------------- message m2: product
    if idx == M*(L+1)-1 % last boundary
        m2_eta = m1_eta + prior_eta;
        m2_Lamb = m1_Lamb + prior_Lamb;
    else
        m2_eta = m1_eta + prior_eta + L_m3_eta(:,idx+1);
        m2_Lamb = m1_Lamb + prior_Lamb + L_m3_Lamb(:,:,idx+1);
    end

    % ----------------------------- message m3: sum
    m2_Sigma = pinv(m2_Lamb); % find the matrix Sigma of m2
    pos = [mod(idx-1,M)+1, mod(idx-1,M)+1+M]; % pos of two variables (real and imag) to be integrated
    % convert m2_eta back to m2_mean to delete columns -> convert back and add zero columns -> get the new m3_eta
    m2_mean = m2_Sigma * m2_eta; % m2_mean
    m2_mean(pos) = []; % set to zero and convert back to eta (see below)
    % convert m2_Lambda back to m2_Sigma to delete rows/columns -> convert back and add zero rows/columns -> get the new m3_Lambda
    m2_Sigma(pos,:) = []; % delete the rows and columns of m2_Sigma
    m2_Sigma(:,pos) = [];
    m3_Lamb = pinv(m2_Sigma); % the dimension-reduced m3_lambda
    m3_Lamb_add0 = [m3_Lamb(1:(pos(1)-1),:); zeros(1,2*M-2); m3_Lamb(pos(1):pos(1)+M-2,:); zeros(1,2*M-2); m3_Lamb(pos(1)+M-1:end,:)]; % insert two all-zero rows to m3_lambda
    m3_Lamb_add0 = [m3_Lamb_add0(:,1:(pos(1)-1)), zeros(2*M,1), m3_Lamb_add0(:,pos(1):pos(1)+M-2), zeros(2*M,1), m3_Lamb_add0(:,pos(1)+M-1:end)]; % insert two all-zero columns to m3_lambda
    m3_eta = m3_Lamb * m2_mean;
    m3_eta_add0 = [m3_eta(1:(pos(1)-1)); 0; m3_eta(pos(1):pos(1)+M-2); 0; m3_eta(pos(1)+M-1:end)];
    % ----------------------------- store m3
    L_m3_eta(:,idx) = m3_eta_add0;
    L_m3_Lamb(:,:,idx) = m3_Lamb_add0;
end

%% ------------------------- BP DECODING
Sum_mu = zeros(2,L);
%     Sum_Sigma = zeros(2,L);
for ii = 1 : L
    % compute (eta,Lamb) for a variable node
    idx = ii * M;
    Res_Eta = etaMat(:,idx) / VarVec(mod(idx-1,M)+1) + prior_eta4 + R_m3_eta(:,idx-1) + L_m3_eta(:,idx+1);
    Res_Lamb = ObserMat4 / VarVec(mod(idx-1,M)+1) + prior_Lamb4 + R_m3_Lamb(:,:,idx-1) + L_m3_Lamb(:,:,idx+1);

    % compute (mu,Sigma) for a variable node
    Res_Sigma = pinv(Res_Lamb);
    Res_mu = Res_Sigma * Res_Eta;

    % compute (mu,Sigma) for the sum
    Sum_mu(:,ii) = [sum(Res_mu(1:M)), sum(Res_mu(M+1:end))]';
%         Sum_Sigma(:,ii) = [sum(sum(Res_Sigma(1:M,1:M))), sum(sum(Res_Sigma(M+1:end,M+1:end)))]';
end

real_sum_seq = [Sum_mu(1,:), Sum_mu(2,:)];
