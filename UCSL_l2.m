function [Z, U_tilde]=CoJLSLR_l2(X_tilde,L,c,alpha,beta,gamma)

num = size(X_tilde,2);
    
%% L_2 norm
%% compute Z
H_tilde = X_tilde*X_tilde'+alpha*eye(size(X_tilde*X_tilde'))+beta*X_tilde*L*X_tilde';
Temp = gamma*L+eye(num,num)-X_tilde'/(H_tilde)*X_tilde;
% tic
% [V1, D1, W1] = eig(Temp);
% toc
% tic
[V, D, W] = eig((Temp+Temp')/2);
% toc
[d,ind] = sort(diag(D),'ascend');%ascend
Ds = D(ind,ind);
%% right columwise eigen vector
% Vs = V(:,ind);
% Y = Vs(:,1:c);%Y'*Y=I_c
% U_M = H_M\X_M*Y;
% U_H = H_H\X_H*Y;
%% left eigen vector
Ws = W(ind,:);
Z = Ws(1:c,:)';%W'*W=I_c
%% compute U
U_tilde = H_tilde\X_tilde*Z;
%% L_21 norm
% maxiter = 1000; 
% epsilon = 1e-4; % Tolerance error
% iter=1; 
% stop = false;
% while ~stop && iter < maxiter+1
%     
%     Q_M = norm21weight(U_M);
%     Q_H = norm21weight(U_H);
%     
%     H_M = X_M*X_M'+beta*Q_M+gamma*X_M*L*X_M';
%     H_H = X_H*X_H'+beta*Q_H+gamma*X_H*L*X_H';
%     Temp = L+alpha*eye(num,num)-alpha*(X_M'/(H_M)*X_M+X_H'/(H_H)*X_H);
%     [~, D, W] = eig(Temp);
% %     [~, D, W] = eig((Temp+Temp')/2);
%     [~,ind] = sort(diag(D),'ascend');%ascend
% 
%     %% left eigen vector
%     Ws = W(ind,:);
%     Y = Ws(1:c,:)';%W'*W=I_c
%     
%     %% compute U
%     Temp_U_M = U_M;
%     Temp_U_H = U_H;
%     U_M = H_M\X_M*Y;
%     U_H = H_H\X_H*Y;
%     
%     r_H(iter) = norm(U_H-Temp_U_H,'fro');
%     r_M(iter) = norm(U_M-Temp_U_M,'fro');
%     
%     %check the convergence conditions
%     if r_H(iter)<epsilon && r_M(iter)<epsilon
% %         stop = true;
%         break;
%     end
%     
%     iter = iter+1;
%     
% end