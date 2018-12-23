videosdir = 'D:\GaitProject\openposeCMU';
avifile = 'qq.mp4'; 
videoSource = vision.VideoFileReader(fullfile(videosdir, avifile) ,'ImageColorSpace','Intensity','VideoOutputDataType','uint8');
peopleDetector = vision.PeopleDetector;
while ~isDone(videoSource)
    frame = step(videoSource);
    [bboxes,scores] = step(peopleDetector,frame);
    if isempty(scores)
        continue;
    end
    frame = insertObjectAnnotation(frame,'rectangle',bboxes,scores);
    figure, imshow(frame)
end

% videosdir='D:\GaitProject\experiment\CASIA\DatasetB\InvVideo';
% experdirbase = './data/tmp';
% avifile = '121-nm-01-180inv.avi';       % CHANGE ME!
% minArea = 3000;     % Minimum area of the BB. Adapt to your dataset.
% aspectRatio = 2.5;    % Aspect ratio between width and height of the BB.
% offset = 0.2;       % Percentage of increase of each dimension of the BB.
% drawBB = true;     % Draw BB? Set to true to show frames with detections
% 
% if ~exist(experdirbase, 'dir')
%    mkdir(experdirbase);
% end
% 
% %% Run it!
% [folder, videoname, ext] = fileparts(avifile);
% BB = [];
% videoSource = vision.VideoFileReader(fullfile(videosdir, avifile) ,'ImageColorSpace','Intensity','VideoOutputDataType','uint8');
% detector = vision.ForegroundDetector(...
%     'NumTrainingFrames', 90, ...
%     'InitialVariance', 70*70, 'NumGaussians', 50,'MinimumBackgroundRatio',0.7); % initial standard deviation of 30
% nFrame = 1;
% while ~isDone(videoSource)
%     % Apply an aperture to normalize the segmentation.
%     frame  = step(videoSource);
%     fgMask = step(detector, frame);
%     
%     % Concatenate BBs of the whole sequence.
%     BBi = fc_getBBWithSegmentation(fgMask, minArea, aspectRatio, offset, nFrame);
%     if ~isempty(BBi)
%         BB = cat(1, BB, BBi);  
%     end
%     nFrame = nFrame + 1;
% end
% release(videoSource);
% videosdir='D:\GaitProject\DataSet\CASIA\DatasetB\videos';
% avifile = [avifile(1:end-7),'.avi'];
% videoSource = vision.VideoFileReader(fullfile(videosdir, avifile) ,'ImageColorSpace','Intensity','VideoOutputDataType','uint8');
% framecount = nFrame-1;
% bbidx = size(BB,1);
% T = regexp({BB.image_path},'\.','split');
% T = str2double([T{:}]);
% T = T(~isnan(T));
% nFrame = 1;
% while ~isDone(videoSource) && bbidx >0
%     frame  = step(videoSource);    
%     if  framecount-nFrame+1 ==T(bbidx)
%         BBi = BB(bbidx);
%         bbidx = bbidx-1;
%         imshow(frame); hold on
%         title(sprintf('Frame %03d', nFrame));
%         rec = [BBi.x, BBi.y, BBi.width, BBi.height];
%         hr = rectangle('Position', rec);
%         set(hr, 'EdgeColor', 'red');
%         set(hr, 'LineWidth', 3);
%         pause(1.0/25);
%     end
%     nFrame = nFrame+1;
%   
%    
% end
% release(videoSource);     
