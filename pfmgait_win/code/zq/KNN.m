function [ Ck ] = KNN( X,C,Xk,K )
%X：训练样本
%C：标签
%Xk：识别样本
%K：取值
size_Xk = size(Xk);
Xk_numbers = size_Xk(2);
size_X = size(X);
X_numbers = size_X(2);
Ck = zeros(1,Xk_numbers);

for i = 1:Xk_numbers
      temp_X = repmat(Xk(:,i),1,X_numbers) ;
      distance = dis(temp_X,X);
     [ distance ,sort_index] =sort(distance);
     label_set = unique(C(sort_index(1:K)));   
     t = histc(C(sort_index(1:K)),label_set);
     [drop id] = max(t);
     Ck(i) = label_set(id);
end

end
%%
%子函数用来计算距离
function  d = dis(x,y) 
   size_x = size(x);
   d= zeros(size_x(2),1);
    for i = 1:size_x(2)
        d(i) = norm(x(:,i)-y(:,i));
    end
%     d(d==0) = 0.0001;
end
