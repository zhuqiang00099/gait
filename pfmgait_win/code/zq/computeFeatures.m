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
type = ['nm'];
degree = num2str((0:18:162)','%03d');
%train��ʾ�ļ����е��ֶ�01,02,03,04��������ѡ������ѵ��������������
train = ['02';'03';'05'];
str = '000-nm-02-000.avi';
filenames = repmat(str,size(subjects,1)*size(degree,1)*size(train,1)*size(type,1),1);
id = 1;
for i = 1:size(subjects,1)
    for j = 1:size(train,1)
        for k = 1:size(degree,1)
            for m = 1:size(type,1)
                 filenames(id,:) = [subjects(i,:) '-' type(m,:) '-' train(j,:) '-' degree(k,:) '.avi'];
                  id = id+1;
            end           
        end
    end
end
for id = 1:length(filenames)
    if ~exist([videosdir,'\',filenames(id,:)],'file')
        error([videosdir,'\',filenames(id,:),'������'])
    end
end

%% ��������,����par���м���
parfor id = 1:length(filenames)
    if exist([experdirbase '\' filenames(id,1:end-4),'_W01_H02.mat'],'file')
        continue;
    end
    computeFeat(videosdir,experdirbase,filenames(id,:));
end

clc,clear;
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
type = ['bg';'cl'];
degree = num2str((0:18:162)','%03d');
%train��ʾ�ļ����е��ֶ�01,02,03,04��������ѡ������ѵ��������������
train = ['01'];
str = '000-nm-02-000.avi';
filenames = repmat(str,size(subjects,1)*size(degree,1)*size(train,1)*size(type,1),1);
id = 1;
for i = 1:size(subjects,1)
    for j = 1:size(train,1)
        for k = 1:size(degree,1)
            for m = 1:size(type,1)
                 filenames(id,:) = [subjects(i,:) '-' type(m,:) '-' train(j,:) '-' degree(k,:) '.avi'];
                  id = id+1;
            end           
        end
    end
end
for id = 1:length(filenames)
    if ~exist([videosdir,'\',filenames(id,:)],'file')
        error([videosdir,'\',filenames(id,:),'������'])
    end
end

%% ��������,����par���м���
parfor id = 1:length(filenames)
    if exist([experdirbase '\' filenames(id,1:end-4),'_W01_H02.mat'],'file')
        continue;
    end
    computeFeat(videosdir,experdirbase,filenames(id,:));
end