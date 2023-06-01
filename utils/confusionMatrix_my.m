function [ M,oa,pa,ua,kappa ] = confusionMatrix_my( label, claMap )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if size(label,2)==1
    label=label';
end
if size(claMap,1)==1
    claMap=claMap';
end

n = length(label);
% thr = 30000;
sec = 10000;

claIdx_lab = unique(label);
claIdx_map = unique(claMap);
M = zeros(length(claIdx_lab), length(claIdx_lab));
        
% if n > thr
maxiter = floor(n/sec);
k=0;
if maxiter>0
    for k = 1:maxiter
        claMap_k = claMap((k-1)*sec+1:k*sec, :);
        label_k = label(:,(k-1)*sec+1:k*sec);

        ind_k = label_k~=0;
        claMap_k = claMap_k.*ind_k;

        label_k(label_k==0)=[];
        claMap_k(claMap_k==0)=[];

        M_k = zeros(length(claIdx_lab), length(claIdx_lab));

        l = length(label_k);

        for i = 1:l
            M_k(claIdx_lab==label_k(i),claIdx_map==claMap_k(i)) = M_k(claIdx_lab==label_k(i),claIdx_map==claMap_k(i)) + 1;
        end

        M = M + M_k;
    end
end

claMap_k = claMap(k*sec+1:end, :);
label_k = label(:, k*sec+1:end);

ind_k = label_k~=0;
claMap_k = claMap_k.*ind_k;

label_k(label_k==0)=[];
claMap_k(claMap_k==0)=[];

M_k = zeros(length(claIdx_lab), length(claIdx_lab));

l = length(label_k);

for i = 1:l
    M_k(claIdx_lab==label_k(i),claIdx_map==claMap_k(i)) = M_k(claIdx_lab==label_k(i),claIdx_map==claMap_k(i)) + 1;
end

M = M + M_k;    

% overall accuracy
oa = sum(diag(M))./sum(M(:))*100;

% producer accuracy(accuracy for each class)
pa = diag(M)./sum(M,2)*100;

% user accuracy(in each classified class, the percentage of correct classified)
ua = diag(M)./sum(M,1)';

% kappa coefficient
po = oa;
pe = sum(sum(M,1).*sum(M,2)')/n^2*100;
kappa = (po-pe)/(100-pe);

end