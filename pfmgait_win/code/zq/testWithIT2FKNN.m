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
if ~exist([experdirbase,'\','dictionary_K=100.mat'],'file')
    error('FV�ֵ䲻����')
end
load([experdirbase,'\','dictionary_K=100']);

if ~exist([experdirbase,'\','Features_K=100,PCAH=256,d=30,s=124,DOWN.mat'],'file')
    error('����������')
end
load([experdirbase,'\','Features_K=100,PCAH=256,d=30,s=124,DOWN']);

%% ����
results = zeros(124,1);
pars = [];

parfor id = 1:length(filenames)
    if ~exist([experdirbase,'\',subjects(id,:),'-',type(1,:),'-',train_sym(1,:),'-',degree(1,:),'_W01_H02.mat'],'file')
        computeFeat(videosdir,experdirbase,filenames(id,:));         
    end
end

testSamples = zeros(256,length(filenames),'single');

for id = 1:length(filenames)
    load([experdirbase,'\',subjects(id,:),'-',type(1,:),'-',train_sym(1,:),'-',degree(1,:),'_W01_H02.mat']);
    matrix_prd2 = fc_calculateFeatsMatrix(detections, [1 2]);
    matrix_prd1 = [matrix_prd2{1,1} matrix_prd2{1,2}];
%     if iscell(matrix) % Several partitions
%         pfm = [];
%        for ixmt = 1:length(matrix)
%              pfm_ = mj_encodeFV(matrix{ixmt}, dictionary{ixmt}, pars);
%              pfm = [pfm; pfm_];
%        end
%    else
%        pfm = mj_encodeFV(matrix, dictionary, pars);
%     end
%    pfm1= mj_encodeFV(matrix_prd1,dictionary{1},pars);
%    pfm2= mj_encodeFV(matrix_prd2{1},dictionary{2},pars);
   pfm3= mj_encodeFV(matrix_prd2{2},dictionary{3},pars);
   
%    pfm = [pfm1;pfm2;pfm3];
    pfm = pfm3;
    pfm = pfm - feats.pcaM;
    pfm = pfm'*feats.pcaP;
    testSamples(:,id) = pfm';
end
results = IT2FKNN(feats.data,feats.labels,testSamples,[1 2 3]);
% results = KNN(feats.data,feats.labels,testSamples,3);
a = results - (1:124);
a = a==0;
sum(a)
