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
type = ['nm'];
degree = num2str((0:18:162)','%03d');
%train表示文件名中的字段01,02,03,04，可以挑选几个来训练，几个来测试
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

clc,clear;
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
type = ['bg';'cl'];
degree = num2str((0:18:162)','%03d');
%train表示文件名中的字段01,02,03,04，可以挑选几个来训练，几个来测试
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