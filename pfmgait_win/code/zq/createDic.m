videosdir = 'D:\GaitProject\DataSet\CASIA\DatasetB\videos';
experdirbase = 'D:\GaitProject\experiment\CASIA\DatasetB\train_feats_nm_000';
%% 检查路径和文件
if ~exist(videosdir,'dir')
    error( '视频文件夹路径不存在')    
end
if ~exist(experdirbase,'dir') 
    mkdir(experdirbase);
end
%CASIA视频文件名由三部分组成：编号，类型[bg,nm,bkgrd,cl]，01/02 , 拍摄角度
subjects = num2str((1:124)','%03d');
type = 'nm';
degree = num2str((0:18:162)','%03d');
%train表示文件名中的字段01,02,03,04，可以挑选几个来训练，几个来测试
train = ['02';'03';'05'];
str = '000-nm-02-000.avi';
filenames = repmat(str,size(subjects,1)*size(degree,1)*size(train,1),1);
id = 1;
for i = 1:size(subjects,1)
    for j = 1:size(train,1)
        for k = 1:size(degree,1)
            filenames(id,:) = [subjects(i,:) '-nm-' train(j,:) '-' degree(k,:) '.avi'];
            id = id+1;
        end
    end
end
for id = 1:length(filenames)
    if ~exist([videosdir,'\',filenames(id,:)],'file')
        error([videosdir,'\',filenames(id,:),'不存在'])
    end
end

%% 计算特征,采用par并行计算
parfor id = 1:length(filenames)
    if exist([experdirbase '\' filenames(id,1:end-4),'_W01_H02.mat'],'file')
        continue;
    end
    computeFeat(videosdir,experdirbase,filenames(id,:));
end

%% FV字典学习
%首先load所有特征
if ~exist([experdirbase,'\','sparse_dictionary_K=100.mat'],'file')     
     lprd_cell1 = 0;
     lprd_cell2 = 0; 
     %预分配内存
     sprintf('字典学习\n预分配内存\n')
     for id = 1:length(filenames)   
        load([experdirbase '\' filenames(id,1:end-4),'_W01_H02.mat']);       
        lprd_cell1 = lprd_cell1 + length(detections{1});
        lprd_cell2 = lprd_cell2 + length(detections{2});
        clear detections;
     end 
     sprintf('预分配内存完毕，cell1: %d\tcell2: %d\n',lprd_cell1,lprd_cell2)
%       matrix = fc_calculateFeatsMatrix(detections, [1 2]);
%         all_matrix{id} = matrix;
     pack;%整理碎片
     prd2_cell1 = zeros(318,lprd_cell1,'single');
     prd2_cell2 = zeros(318,lprd_cell2,'single');
   
     id1 = 1;    %赋值到哪一列
     id2 = 1;   
     sprintf('特征赋值:\n')
     for id = 1:length(filenames)   
        load([experdirbase '\' filenames(id,1:end-4),'_W01_H02.mat']);
        cell1 = [detections{1}.feats];
        cell2 =[ detections{2}.feats];
        len1 = length(cell1);
        len2 = length(cell2);      
        prd2_cell1(:,id1:(id1+len1-1)) = cell1;
        prd2_cell2(:,id2:(id2+len2-1))= cell2;
        id1 = id1+len1;
        id2 = id2+len2;        
        clear detections;
        sprintf('进度>>>>>>>>%02d%%\n',id/length(filenames)*100)
     end    
 
     fv2 = FV(prd2_cell1,100);     
     fv2.clearData();
     fv3 = FV(prd2_cell2,100);     
     fv3.clearData();
     prd1 = [prd2_cell1,prd2_cell2];
     clear prd2_cell1 prd2_cell2;
     fv1 = FV(prd1,100);     
     fv1.clearData();
     clear prd1;
     dictionary = {fv1,fv2,fv3};
     save([experdirbase,'\','sparse_dictionary_K=100'],'dictionary');
else
    load([experdirbase,'\','sparse_dictionary_K=100']);
end


%% 计算所有PFM向量
 trainSamples = zeros(2*100*318*1,length(filenames),'single');
 pars =[];
if ~exist([experdirbase,'\','Features_K=100,PCAH=3720,d=sparse,s=124,FULL.mat'],'file')
         parfor id = 1:length(filenames)
            d=load([experdirbase '\' filenames(id,1:end-4),'_W01_H02.mat']);
            if isempty(d.detections)
                trainSamples(:,id) = 0;
                continue;
            end
%             matrix_prd2 = fc_calculateFeatsMatrix(detections, [1 2]);
%                 matrix_down = [d.detections{2}.feats];
%              matrix_up = [d.detections{1}.feats];
%             matrix_prd1 = [matrix_prd2{1,1} matrix_prd2{1,2}];
                matrix_prd1 = [d.detections{1}.feats,d.detections{2}.feats];
%             if iscell(matrix) % Several partitions
%                pfm = [];
%                for ixmt = 1:length(matrix)
%                    pfm_ = mj_encodeFV(matrix{ixmt}, dictionary{ixmt}, pars);
%                    pfm = [pfm; pfm_];
%                end
%             else
%                 pfm = mj_encodeFV(matrix, dictionary, pars);
%             end
            pfm1= mj_encodeFV(matrix_prd1,dictionary{1},pars);
%              pfm2= mj_encodeFV(matrix_up,dictionary{2},pars);
%             pfm3= mj_encodeFV(matrix_down,dictionary{3},pars);
%             pfm = [pfm1;pfm2;pfm3];
            pfm = pfm1;
            trainSamples(:,id) = pfm;
        end
%PCA     256 
       [pcaM_, scores,latent] = princomp(trainSamples','econ');    
       latent = latent./sum(latent);
       latent = cumsum(latent);
       save('latent','latent');
       pcaP = pcaM_(:,1:end);   
       pcaM = mean(trainSamples,2);
       trainSamples= scores(:,1:end)'; 
       feats.pcaP = pcaP;
       feats.pcaM = pcaM;
       feats.data = trainSamples;
       labels = repmat(1:124,1,size(train,1)*size(degree,1));
       labels = sort(labels);
       labels = labels';
       feats.labels = labels;
       save([experdirbase '\' 'Features_K=100,PCAH=3720,d=sparse,s=124,FULL.mat'],'feats');
       clear trainSamples;
else
    load([experdirbase '\' 'Features_K=100,PCAH=3720,d=sparse,s=124,FULL.mat']);    
end
%% 训练
isSVM = true;
if ~exist([experdirbase,'\','svm_model_dictionary_K=100,PCAH=3720,d=sparse,s=124,FULL.mat'],'file') && isSVM
       conf.svm.C = 10;
       conf.svm.biasMultiplier = 1;
       conf.normalize = false;      
       kindclassif = 'svmlin';
       [model, acc, acc_test_pc] = mj_trainMultiClass(feats.data', feats.labels, kindclassif, conf);
       model.pcaP = feats.pcaP;
       model.pcaM = feats.pcaM;
       save([experdirbase,'\','svm_model_dictionary_K=100,PCAH=3720,d=sparse,s=124,FULL'],'model');
else
      load([experdirbase,'\','svm_model_dictionary_K=100,PCAH=3720,d=sparse,s=124,FULL']);
end

%% 分类
test_video = '030-bg-02-054.avi';
detections = computeFeat(videosdir,experdirbase,test_video);
% matrix_prd2 = fc_calculateFeatsMatrix(detections, [1 2]);
% matrix_prd1 = [matrix_prd2{1,1} matrix_prd2{1,2}];
matrix_prd1 = [detections{1}.feats,detections{2}.feats];
% matrix_down = [detections{2}.feats];
% matrix_up = [detections{1}.feats];
%  if iscell(matrix) % Several partitions
%        pfm = [];
%        for ixmt = 1:length(matrix)
%              pfm_ = mj_encodeFV(matrix{ixmt}, dictionary{ixmt}, pars);
%              pfm = [pfm; pfm_];
%         end
%   else
%        pfm = mj_encodeFV(matrix, dictionary, pars);
%  end 
 pfm1= mj_encodeFV(matrix_prd1,dictionary{1},pars);
%  pfm2= mj_encodeFV(matrix_up,dictionary{2},pars);
%  pfm3= mj_encodeFV(matrix_down,dictionary{3},pars);
% pfm = [pfm1;pfm2;pfm3];
pfm = pfm1;
pfm = pfm - model.pcaM;
pfm = pfm'*model.pcaP;
[vidEstClass, svmscores, acc_test, acc_test_pc] = mj_classifyMultiClass(pfm, [], model);




