function [ rate,C, errors,e_id] = RANKN( scores ,N,labels)
%输入：scores:每个样本的分数,ndims x samples，每一列一个样本
%           N: 前几
%           labels:对应标签 samples x 1
%输出：C 每个样本对应的标签
% 
N_samples = size(scores,2);
[scores,id] = sort(scores,'descend');
C = id(1:N,:);
labels = repmat(labels,N,1);
if N>1
     e_id = find(sum((C-labels)==0)==0);
else
     e_id = find(C-labels);
end
errors = length(e_id);
rate = 1-errors / N_samples;
end

