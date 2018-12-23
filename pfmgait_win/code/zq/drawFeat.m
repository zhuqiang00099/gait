videosdir = 'D:\GaitProject\DataSet\CASIA\DatasetB\videos';
experdirbase = 'D:\GaitProject\experiment\CASIA\DatasetB\train_feats_nm_000';

 load([experdirbase '\022-bg-01-090\022-bg-01-090','.wFlowT0C1.features.mat']);  
 mj_displayDenseFeatsOnVideo(F,[videosdir,'\022-bg-01-090.avi']);