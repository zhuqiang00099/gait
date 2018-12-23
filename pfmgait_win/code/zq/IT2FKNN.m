function [ C_Xk ] = IT2FKNN( X,C,Xk,init_Kset)
%X������
%C��������Ӧ�ķ���
%Xk������������
%init_Kset����ʼ��K��
%C_Xk:����������
%%
%��ʼ��ѵ������X�������ȣ���ѡ���K�������ģ�������ⲿ�����ݿ��Դ洢����
size_X = size(X);
X_numbers = size_X(2);
init_Uset = zeros(X_numbers,length(unique(C)),length(init_Kset));%����X��������,�±�ֱ�Ϊ������k,���c,��ӦKֵ
max_K = max(init_Kset); 

for j = 1:X_numbers
           distance = dis(repmat(X(:,j),1,X_numbers),X);           
           [distance,sort_index] = select_min(distance,max_K+1);  %��ǰK��
           for i = 1:length(init_Kset)
                  for k = 1:init_Kset(i)
                      init_Uset(j,C(sort_index(k+1)),i) = init_Uset(j,C(sort_index(k+1)),i)+0.49/init_Kset(i);
                  end  
                  init_Uset(j,C(j),i) =  init_Uset(j,C(j),i) +0.51;                 
           end 
           for c = 1:length(unique(C))%ȥ�ظ�������
               temp_U =unique(init_Uset(j,c,:));                     
               init_Uset(j,c,:) = 0;
               init_Uset(j,c,1:length(temp_U)) = temp_U;               
           end           
 end
%  reshape(init_Uset(1,:,:),length(unique(C)),length(init_Kset))
% init_Uset(1,1,:)
% init_Uset(1,2,:)

%%
%���������������������
size_Xk = size(Xk);
Xk_numbers = size_Xk(2);
U_set = zeros(Xk_numbers,length(unique(C)));
K =randperm(length(init_Kset));%�������һ��K
K = init_Kset(K(1));
for i = 1:Xk_numbers%������ʶ������
    distance = dis(repmat(Xk(:,i),1,X_numbers),X);
    [distance,sort_index] = select_min(distance,K); 
       %�൱�������ȹ�ʽ�У�m = 2��������Ҫ���������    
    distance = 1./distance;
    den = sum(distance); %�����ȵķ�ĸ
    num = zeros(1,K); %����
    for j = 1:length(unique(C))%Ϊÿһ��������������
        temp_U = init_Uset(sort_index,j,:); 
        temp_U = reshape(temp_U,K,length(init_Kset)); %���γ� K�У�init_Kset�У�ÿһ�б�ʾͬһ������������
%         temp_U = sort(temp_U,2,'descend'); 
        
        for k = 1:K   %�����
             if sum(temp_U(k,:)) == 0
                 num(k) = 0;
             else
                 num(k) = sum(temp_U(k,:))/sum(temp_U(k,:)>0);
             end
              
        end   
        num = num(:).*distance(:);
        U_set(i,j) = sum(num)/den;
        num = 0;
    end
end
%%
%���������ȷ���
C_Xk = zeros(1,Xk_numbers);%��ʼ���������
for i = 1:Xk_numbers
    [u_max,C_Xk(i) ]= max(U_set(i,:));
end

% U_set
end
%%
%�Ӻ��������������
function  d = dis(x,y) 
   size_x = size(x);
   d= zeros(1,size_x(2));
    for i = 1:size_x(2)
        d(i) = norm(x(:,i)-y(:,i));
    end
    d(d==0) = 0.0001;
end
function [data,index] = select_min(X,K)  
       size_X = size(X);
      data = zeros(size_X(1),K);
      index = zeros(K,1);
      for i = 1:K
           [data(:,i),index(i)] = min(X);
           X(:,index(i)) = inf;
      end
end

