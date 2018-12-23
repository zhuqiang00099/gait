function  computeFeat( video_path,out_path)
%video_path 视频路径
%outdir 特征文件输出目录
tooldir = '.\tools\';
idt_tool = [tooldir,'iDT_release_zq'];
gzip_tool = [tooldir,'gzip'];
[folder, videoname, ext] = fileparts(video_path);
temp_gz = [videoname,'.feats.gz'];
args = ' -W 2';
system([idt_tool,' ',video_path,args,' | ' ,gzip_tool ,' > ',temp_gz]);
gunzip(temp_gz,'.');
F = loadDenseFeatFile([videoname,'.feats']);
save(out_path, 'F');
system(['rm -f ',temp_gz]);
system(['rm -f ',videoname,'.feats']);

end

