function [ pfm ] = getPFM( videosdir,experdirbase,avifile,K_,doPCA)
%从一个视频中获得PFM特征
%filename:视频路径


%% setp1:检测人
disp('**  step1:detect people in a video sequence **');

minArea = 1000;     % Minimum area of the BB. Adapt to your dataset.
aspectRatio = 3;    % Aspect ratio between width and height of the BB.
offset = 0.2;       % Percentage of increase of each dimension of the BB.
drawBB = true;     % Draw BB? Set to true to show frames with detections

if ~exist(experdirbase, 'dir')
   mkdir(experdirbase);
end

% [folder, videoname, ext] = fileparts(avifile);
BB = [];
videoSource = vision.VideoFileReader(fullfile(videosdir, avifile) ,'ImageColorSpace','Intensity','VideoOutputDataType','uint8');
detector = vision.ForegroundDetector(...
    'NumTrainingFrames', 40, ...
    'InitialVariance', 30*30, 'NumGaussians', 10); % initial standard deviation of 30
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
% output = fullfile(experdirbase, [videoname '-bb.mat']);
% save(output, 'BB');
% fprintf('Written file %s. \n', videoname);

%% setp2:跟踪
disp('** step2: generate people tracks from detections **');
% detfile = 'p008-n05-bb.mat';       % CHANGE ME!
videofile = avifile;
minFrames = 0;      % Minimum number of frames that a track must have.
detectionOffset = 3; % Offset added to every BB score. Used to normalize
% scores between different detectors.
pars.time_stdv = 2; % Parameter used for tracking.
pars.cp_thresh = 0.2; % Parameter used for tracking.
metric = 'chisq'; % Metric for joining tracks.
joinThreshold = 15000; % Threshold for joining tracks.
nBeans = 16; % Number of components of the histograms.

% det = load(fullfile(detectionsdir, detfile));
det = BB;
% Building the matrix for the tracking process.
detections = zeros(10, length(det));
T = regexp(det(1).image_path, '/', 'split');
for j=1:length(det)
    T = regexp(det(j).image_path, '/', 'split');
    T = regexp(T{1, length(T)}, '\.', 'split');
    
    detections(1, j) = str2double(T{1, 1});
    detections(2, j) = det(j).x;
    detections(3, j) = det(j).y;
    detections(4, j) = det(j).width;
    detections(5, j) = det(j).height;
    detections(6, j) = 1;
    detections(7, j) = det(j).score;
    detections(8, j) = 1;
    detections(9, j) = 2;
    detections(10, j) = 1;
end

% Remove nested detections?
detections = mj_filterInsideDets(detections);
shot = [detections(1, 1) detections(1, size(detections, 2))];

% Tracking.
addpath('./libs/ubtrack/matlab');
tracks = track(detections, shot, pars);
scores = fc_evaluateTracks(tracks, minFrames, detectionOffset);

% Interpolate tracks.
for j=1:length(tracks)
    tracks(j).D = InterpolateTrack(tracks(j).D);
    [total_x, avg_x, total_y, avg_y] = fc_computeTrackDisplacement(tracks(j).D);
    if (abs(total_x) + abs(total_y)) < 1
        scores(j) = 0;
    end
end

% Smooth tracks.
for trix = 1:length(tracks)
    tracks(trix).D(2:5,:) = me_smoothBB(tracks(trix).D(1,:)-tracks(trix).D(1,1),tracks(trix).D(2:5,:),3,2);
end

scoredTracks = struct('tracks', tracks, 'scores', scores);

if length(tracks) > 1
    scoredTracks = fc_joinTracks(fullfile(videosdir, videofile), scoredTracks, nBeans, metric, joinThreshold);
end

% Save results.
detections = scoredTracks;
% output = fullfile(experdirbase, outputname);
% save(output, 'detections');
% fprintf('Written file %s. \n', outputname);

%% setp3:获取DT
disp('** This demo shows how to extract dense tracklets from a video sequence **');
[folder, videoname, ext] = fileparts(avifile);
extrapars.binpath = './PFMtool'; % Set your path here
extrapars.binparams = '-T 0 -C 1';
outdensefile = mj_computeDenseFeats(videosdir, experdirbase, videoname, '', extrapars, false, false, false);

if ~isempty(outdensefile) && exist(outdensefile, 'file')
   % Convert to mat file
   matdtfile = mj_gzfile2mat(outdensefile);
else
   matdtfile = '';
   warning('Something went wrong...','PFM:noOutputDenseTracksFile');
end

disp(['Output saved to :', matdtfile]);

%% step4:过滤DT
disp('** step4: filter out dense tracklets given people detections **');
dtdir = './data';
% trackdir = './data/tmp';
dtfile = 'p008-n05.wFlowT0C1.features.mat'; %'p005-n05.mat';       % CHANGE ME!
% trackfile = '008-n-05_fb_tracks.mat';       % CHANGE ME!

grid.horizontal = 1; % Array that contains the percentage limits of the
% parts in the horizontal axis.
grid.vertical = [0.5 0.5]; % Array that contains the percentage limits of the parts in the
% vertical axis.
params.offset = 0; % Percentage offset of the detection.
threshold = 50; % Minimum score of a track.

% Loading features.
features = load(fullfile(dtdir, dtfile));
features = features.F;

% Loading tracks.
% tracks = load(fullfile(trackdir, trackfile));
scores = detections.scores;
tracks = detections.tracks;

% Cleaning and saving tracks.
allFeatures = [];
for i=1:length(tracks)
    fprintf('score = %.2f\n', scores(i));
    if scores(i) > threshold
        finalFeatures = fc_fitFeatures(features, tracks(i), grid, params);
        allFeatures = [allFeatures ; finalFeatures];
    end
end

% Save results.
detections = allFeatures;
%% step5:
disp('** step5: pfm. **');
disp('Computing FV dictionary')
dictionary = FV([detections{1,1}.feats detections{2,1}.feats],K_,doPCA);
dictionary.clearData();
disp('Computing PFM descriptor...');
% Define encoding parameters
pars = []; % Default

% Convert to cell-array of DCS features
matrix = fc_calculateFeatsMatrix(detections, [1 2]);

% Fisher Vector encoding of DCS features: 
if iscell(matrix) % Several partitions
   pfm = [];
   for ixmt = 1:length(matrix)
      pfm_ = mj_encodeFV(matrix{ixmt}, dictionary, pars);
      pfm = [pfm; pfm_];
   end
else
   pfm = mj_encodeFV(matrix, dictionary, pars);
end

disp(size(pfm))
clear detections

end

