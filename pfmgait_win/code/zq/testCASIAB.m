videosdir = 'D:\GaitProject\DataSet\CASIA\DatasetB\videos';
experdirbase = 'D:\GaitProject\experiment\CASIA\DatasetB\train_feats_nm_000';
%% ���·�����ļ�
if ~exist(videosdir,'dir')
    error( '��Ƶ�ļ���·��������')    
end
if ~exist(experdirbase,'dir') 
    mkdir(experdirbase);
end
%CASIA��Ƶ�ļ�������������ɣ���ţ�����[bg,nm,bkgrd,cl]��01/02 , ����Ƕ�
subjects = num2str((1:124)','%03d');
type = repmat('cl',length(subjects),1);
degree = repmat('162',length(subjects),1);
%train��ʾ�ļ����е��ֶ�01,02,03,04��������ѡ������ѵ��������������
train_sym = repmat('01',length(subjects),1);%02����
split_sym = repmat('-',length(subjects),1);
avi_sym = repmat('.avi',length(subjects),1);
filenames = [subjects,split_sym,type,split_sym,train_sym,split_sym,degree,avi_sym];
for id = 1:length(filenames)
    if ~exist([videosdir,'\',filenames(id,:)],'file')
        error([videosdir,'\',filenames(id,:),'������'])
    end
end
%% load dic
if ~exist([experdirbase,'\','sparse_dictionary_K=100.mat'],'file')
    error('FV�ֵ䲻����')
end
load([experdirbase,'\','sparse_dictionary_K=100']);
if ~exist([experdirbase,'\','svm_model_dictionary_K=100,PCAH=3720,d=sparse,s=124,FULL.mat'],'file')
    error('SVMģ�Ͳ�����')
end
load([experdirbase,'\','svm_model_dictionary_K=100,PCAH=3720,d=sparse,s=124,FULL']);
%% ����
results = zeros(124,1);
r_scores = zeros(124);
pars = [];
parfor id = 1:length(filenames)
    if ~exist([experdirbase,'\',subjects(id,:),'-',type(1,:),'-',train_sym(1,:),'-',degree(1,:),'_W01_H02.mat'],'file')
        computeFeat(videosdir,experdirbase,filenames(id,:));         
    end
end
parfor id = 1:length(filenames)
    d=load([experdirbase,'\',subjects(id,:),'-',type(1,:),'-',train_sym(1,:),'-',degree(1,:),'_W01_H02.mat']);
%     matrix_prd2 = fc_calculateFeatsMatrix(detections, [1 2]);
%     matrix_prd1 = [matrix_prd2{1,1} matrix_prd2{1,2}];
        matrix_prd1 = [d.detections{1}.feats,d.detections{2}.feats];
%         matrix_up = [d.detections{1}.feats];
%        matrix_down = [d.detections{2}.feats];
%     if iscell(matrix) % Several partitions
%         pfm = [];
%        for ixmt = 1:length(matrix)
%              pfm_ = mj_encodeFV(matrix{ixmt}, dictionary{ixmt}, pars);
%              pfm = [pfm; pfm_];
%        end
%    else
%        pfm = mj_encodeFV(matrix, dictionary, pars);
%     end
   pfm1= mj_encodeFV(matrix_prd1,dictionary{1},pars);
%    pfm2= mj_encodeFV(matrix_up,dictionary{2},pars);
%    pfm3= mj_encodeFV( matrix_down,dictionary{3},pars);
   
%    pfm = [pfm1;pfm2;pfm3];
    pfm = pfm1;
    pfm = pfm - model.pcaM;
    pfm = pfm'*model.pcaP;
   [vidEstClass, svmscores, acc_test, acc_test_pc] = mj_classifyMultiClass(pfm, [], model);
   results(id) = vidEstClass;
   r_scores(:,id) = svmscores;
end
[rate,C,errors,e_id] =  RANKN(r_scores,1,1:124);
% save(['rup=30-' type(1,:) '-01-' degree(1,:)],'C');
rate
