files = [repmat('r-cl-01-',10,1) num2str((0:18:162)','%03d')];
map = zeros(1,124);
for i = 1:10
    if ~exist([files(i,:) '.mat'],'file')
        continue;
    end
    load(files(i,:));
    a = (C-(1:124))==0;
    map(a==0) = map(a==0)+1;
end
map = (map>2).*map;
[map,id] = sort(map,'descend');
map = map(map>0);
id = id(1:length(map));
bar(map,'r')
set(gca,'XTick',1:length(id))
set(gca,'XTickLabel',num2str(id'))
xlabel('错误视频编号')
ylabel('累加错误次数')
