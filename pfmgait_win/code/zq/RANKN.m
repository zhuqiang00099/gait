function [ rate,C, errors,e_id] = RANKN( scores ,N,labels)
%���룺scores:ÿ�������ķ���,ndims x samples��ÿһ��һ������
%           N: ǰ��
%           labels:��Ӧ��ǩ samples x 1
%�����C ÿ��������Ӧ�ı�ǩ
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

