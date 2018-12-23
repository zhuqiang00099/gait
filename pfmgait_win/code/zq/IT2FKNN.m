function [ C_Xk ] = IT2FKNN( X,C,Xk,init_Kset)
%X：样本
%C：样本对应的分类
%Xk：待分类样本
%init_Kset：初始化K集
%C_Xk:输出样本类别
%%
%初始化训练样本X的隶属度，即选多个K构造二型模糊集，这部分数据可以存储起来
size_X = size(X);
X_numbers = size_X(2);
init_Uset = zeros(X_numbers,length(unique(C)),length(init_Kset));%样本X的隶属度,下标分别为：样本k,类别c,对应K值
max_K = max(init_Kset); 

for j = 1:X_numbers
           distance = dis(repmat(X(:,j),1,X_numbers),X);           
           [distance,sort_index] = select_min(distance,max_K+1);  %挑前K个
           for i = 1:length(init_Kset)
                  for k = 1:init_Kset(i)
                      init_Uset(j,C(sort_index(k+1)),i) = init_Uset(j,C(sort_index(k+1)),i)+0.49/init_Kset(i);
                  end  
                  init_Uset(j,C(j),i) =  init_Uset(j,C(j),i) +0.51;                 
           end 
           for c = 1:length(unique(C))%去重复隶属度
               temp_U =unique(init_Uset(j,c,:));                     
               init_Uset(j,c,:) = 0;
               init_Uset(j,c,1:length(temp_U)) = temp_U;               
           end           
 end
%  reshape(init_Uset(1,:,:),length(unique(C)),length(init_Kset))
% init_Uset(1,1,:)
% init_Uset(1,2,:)

%%
%计算待分类样本的隶属度
size_Xk = size(Xk);
Xk_numbers = size_Xk(2);
U_set = zeros(Xk_numbers,length(unique(C)));
K =randperm(length(init_Kset));%随机生成一个K
K = init_Kset(K(1));
for i = 1:Xk_numbers%遍历待识别样本
    distance = dis(repmat(Xk(:,i),1,X_numbers),X);
    [distance,sort_index] = select_min(distance,K); 
       %相当于隶属度公式中，m = 2，否则需要重新算距离    
    distance = 1./distance;
    den = sum(distance); %隶属度的分母
    num = zeros(1,K); %分子
    for j = 1:length(unique(C))%为每一个类别计算隶属度
        temp_U = init_Uset(sort_index,j,:); 
        temp_U = reshape(temp_U,K,length(init_Kset)); %变形成 K行，init_Kset列，每一行表示同一样本的隶属度
%         temp_U = sort(temp_U,2,'descend'); 
        
        for k = 1:K   %算分子
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
%根据隶属度分类
C_Xk = zeros(1,Xk_numbers);%初始化输出分类
for i = 1:Xk_numbers
    [u_max,C_Xk(i) ]= max(U_set(i,:));
end

% U_set
end
%%
%子函数用来计算距离
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

