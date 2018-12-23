% File: demodetect.m
% Detects people in video sequences.
%

disp('** This demo shows how to detect people in a video sequence **');

%% Configuration
% videosdir = './data';
videosdir = 'D:\GaitProject\DataSet\CASIA\DatasetB\videos';
% videosdir='D:\GaitProject\experiment\CASIA\DatasetB\InvVideo';
experdirbase = './data/tmp';
avifile = '121-nm-01-180.avi';       % CHANGE ME!
minArea = 3000;     % Minimum area of the BB. Adapt to your dataset.
aspectRatio = 2.5;    % Aspect ratio between width and height of the BB.
offset = 0.2;       % Percentage of increase of each dimension of the BB.
drawBB = true;     % Draw BB? Set to true to show frames with detections

if ~exist(experdirbase, 'dir')
   mkdir(experdirbase);
end

%% Run it!
[folder, videoname, ext] = fileparts(avifile);
BB = [];
videoSource = vision.VideoFileReader(fullfile(videosdir, avifile) ,'ImageColorSpace','Intensity','VideoOutputDataType','uint8');
detector = vision.ForegroundDetector(...
    'NumTrainingFrames', 90, ...
    'InitialVariance', 70*70, 'NumGaussians', 50,'MinimumBackgroundRatio',0.8); % initial standard deviation of 30
nFrame = 1;
while ~isDone(videoSource)
    % Apply an aperture to normalize the segmentation.
    frame  = step(videoSource);
    fgMask = step(detector, frame);
    
    % Concatenate BBs of the whole sequence.
    BBi = fc_getBBWithSegmentation(fgMask, minArea, aspectRatio, offset, nFrame);
    if ~isempty(BBi)
        BB = cat(1, BB, BBi);
        % Draw BB
        if drawBB
            imshow(frame); hold on
            title(sprintf('Frame %03d', nFrame));
            rec = [BBi.x, BBi.y, BBi.width, BBi.height];
            hr = rectangle('Position', rec);
            set(hr, 'EdgeColor', 'red');
            set(hr, 'LineWidth', 3);
            pause(1.0/25);
        end
    end
    nFrame = nFrame + 1;
end

release(videoSource);
output = fullfile(experdirbase, [videoname '-bb.mat']);
save(output, 'BB');
fprintf('Written file %s. \n', videoname);
