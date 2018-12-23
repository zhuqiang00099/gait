function trainSparseFVDict()
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
%% FV
if ~exist([experdirbase,'\','dictionary_K=100,sparase=5.mat'],'file')     
    %将filename随机分组
    if ~exist([experdirbase,'\KmeansPoints\','temp_random.mat'],'file')    
        n = 100;
        m = mod(size(filenames,1),n);
        id = randperm(size(filenames,1));
        groups = filenames(id(1:end-m),:);
        remainder = filenames(id(end-m+1:end),:);
        countsPerGroup =size(groups,1)/n;
        save([experdirbase,'\KmeansPoints\','temp_random.mat'],'n','m','id','groups','remainder','countsPerGroup');
    else
        load([experdirbase,'\KmeansPoints\','temp_random.mat']);
    end   
    groupNumber = n;
    if ~isempty(remainder)
       %remainder
       groupNumber = n+1;
       if ~exist([experdirbase,'\KmeansPoints\',sprintf('group%d_kpoints',n+1),'.mat'],'file')
           getKmeansPoints(remainder,experdirbase,sprintf('group%d',n+1));           
       end
    end 
   parfor i =1:n
       fprintf('--------process group%d--------------\n',i);
       if ~exist([experdirbase,'\KmeansPoints\',sprintf('group%d_kpoints',i),'.mat'],'file')
           getKmeansPoints(groups((i-1)*countsPerGroup+1:i*countsPerGroup,:),experdirbase,sprintf('group%d',i));
       end
   end
   
   lenUP = 0;
   lenDOWN = 0;
   lenFULL = 0;
   for i = 1:groupNumber
        load([experdirbase,'\KmeansPoints\',sprintf('group%d_kpoints',i),'.mat']);
        lenUP = lenUP + size(conf.up_C,2);
        lenDOWN = lenDOWN+size(conf.down_C,2);
   end
   featsUP = zeros(318,lenUP,'single');
   featsDOWN = zeros(318,lenDOWN,'single');
   featsFULL = zeros(318,lenFULL,'single');
   id1=1;
   id2=1;
   id3=1;
   for i = 1:groupNumber
       load([experdirbase,'\KmeansPoints\',sprintf('group%d_kpoints',i),'.mat']);
       up = conf.up_C;
       down = conf.down_C;
       full = conf.full_C;
       len1 = size(up,2);
       len2 = size(down,2);
       len3 = size(full,2);
       featsUP(:,id1:(id1+len1-1)) = up;
       featsDOWN(:,id2:(id2+len2-1))= down;
       featsFULL(:,id3:(id3+len3-1)) = full;
       id1 = id1+len1;
       id2 = id2+len2;
       id3 = id3+len3;
   end

     fprintf('-----start train dict up------\n');
     fv2 = FV(featsUP,100);     
     fv2.clearData();
     fprintf('-----start train dict down------\n');
     fv3 = FV(featsDOWN,100);     
     fv3.clearData();  
     fprintf('-----start train dict full------\n');
     fv1 = FV(featsFULL,100);     
     fv1.clearData(); 
     dictionary = {fv1,fv2,fv3};
     save([experdirbase,'\','dictionary_K=100,sparase=5'],'dictionary');
end
end

function points = getKmeansPoints(group,experdirbase,name)
     %remainder
       FeatsNumberCell1 = 0;
       FeatsNumberCell2 = 0;
       for j = 1:size(group,1)
           d= load([experdirbase '\' group(j,1:end-4),'_W01_H02.mat']);
           fprintf('-----%s----\n',group(j,1:end-4));
           if isempty(d.detections) || length(d.detections)<2
               continue;
           end
           FeatsNumberCell1 = length(d.detections{1})+FeatsNumberCell1;
           FeatsNumberCell2 = length(d.detections{2})+FeatsNumberCell2;
       end
       FeatsCell1 = zeros(318,FeatsNumberCell1,'single');
       FeatsCell2 = zeros(318,FeatsNumberCell2,'single');
       id1 = 1;    %赋值到哪一列
       id2 = 1;
       for j = 1:size(group,1)
           d= load([experdirbase '\' group(j,1:end-4),'_W01_H02.mat']);
           if isempty(d.detections) || length(d.detections)<2
               continue;
           end
           cell1 = [d.detections{1}.feats];
           cell2 = [d.detections{2}.feats];
           len1 = size(cell1,2);
           len2 = size(cell2,2);
           FeatsCell1(:,id1:(id1+len1-1)) = cell1;
           FeatsCell2(:,id2:(id2+len2-1))= cell2;
           id1 = id1+len1;
           id2 = id2+len2;
       end
       up_C= kmeans(FeatsCell1,FeatsNumberCell1*0.05,'verbose','distance','l1','initialization','PLUSPLUS',...
           'Algorithm','ANN','NumRepetitions',1);
     
       down_C = kmeans(FeatsCell2,FeatsNumberCell2*0.05,'verbose','distance','l1','initialization','PLUSPLUS',...
           'Algorithm','ANN','NumRepetitions',1);
      
       full_C = kmeans([FeatsCell1,FeatsCell2],(FeatsNumberCell2+FeatsNumberCell1)*0.05,'verbose','distance','l1','initialization','PLUSPLUS',...
           'Algorithm','ANN','NumRepetitions',1);
       conf.up_C = up_C;
       conf.down_C= down_C;
       conf.full_C  = full_C ;
       save([experdirbase,'\KmeansPoints\',name,'_kpoints.mat' ],'conf');
       points = conf;
       fprintf('-------%sKmeansPoints\\%s_kpoints.mat has been saved---------\n',experdirbase,name);
      
   
end




