function F = loadDenseFeatFile(densefile)
% F = loadDenseFeatFile(densefile)
% Load to a mat variable the data included in the text file generated by
% Jain's software
%
% Input: 
%  - densefile: path to file
%
% Output:
%  - F: struct with fields
%    .frix: frame number
%    .mean: mean position
%    .var: variance of position
%    .len
%    .scale
%    .feats: features
%
% (c) MJMJ/2013

F = [];

if ~exist(densefile, 'file')
   error(['File ' densefile ' does not exist.']);
end

%% Open file
fid = fopen(densefile, 'rt');
if fid < 1
   error('Cannot open file.');
end

%% Read data
ix = 0;
stop = false;
while ~stop
   
   linea = fgetl(fid);
   
   if linea == -1 % EOF
      stop = true;
   else
      ix = ix +1 ;
      data = sscanf(linea, '%f', inf);
      
      F(ix).frix = int16(data(1));
      F(ix).mean = single(data(2:3));
      F(ix).var = single(data(4:5));
      F(ix).len = single(data(6));
      F(ix).scale = single(data(7));
      F(ix).spatio = single(data(8:10));
      F(ix).feats = single(data(11:end));
   end
   
end % while

%% Close file
fclose(fid);