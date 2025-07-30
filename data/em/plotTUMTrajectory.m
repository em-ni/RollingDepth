function plotTUMTrajectory(tumFilePath)
% plotTUMTrajectory Reads a TUM trajectory file and plots the 3D trajectory.
%
%   plotTUMTrajectory(tumFilePath)
%
%   Input:
%       tumFilePath - Path to the TUM trajectory file (e.g., 'gt.txt').

% --- Argument Validation ---
if nargin < 1
    error('Usage: plotTUMTrajectory(tumFilePath)');
end
if ~ischar(tumFilePath) && ~isstring(tumFilePath)
    error('Input tumFilePath must be a character array or string.');
end
if ~exist(tumFilePath, 'file')
    error('File not found: %s', tumFilePath);
end

% --- Read TUM File ---
fprintf('Reading TUM trajectory file: %s\n', tumFilePath);
fileID = fopen(tumFilePath, 'r');
if fileID == -1
    error('Cannot open file for reading: %s', tumFilePath);
end

% Read data, skipping header lines (lines starting with '#')
dataLines = {};
tline = fgetl(fileID);
while ischar(tline)
    if ~startsWith(strtrim(tline), '#')
        dataLines{end+1,1} = tline; %#ok<AGROW>
    end
    tline = fgetl(fileID);
end
fclose(fileID);

if isempty(dataLines)
    fprintf('No data found in the TUM file (after skipping headers).\n');
    return;
end

% Parse data
numRows = length(dataLines);
positions = zeros(numRows, 3); % tx, ty, tz

for i = 1:numRows
    try
        % TUM format: timestamp tx ty tz qx qy qz qw
        scanResult = sscanf(dataLines{i}, '%f %f %f %f %f %f %f %f');
        if numel(scanResult) >= 4 % Ensure at least timestamp and x,y,z positions are present
            positions(i, :) = scanResult(2:4)'; % tx, ty, tz
        else
            fprintf('Warning: Skipping malformed line %d: %s\n', i, dataLines{i});
        end
    catch ME
        fprintf('Warning: Error parsing line %d: %s. Error: %s\n', i, dataLines{i}, ME.message);
    end
end

% Remove rows that might have failed parsing (all zeros) if they were initialized
% and not filled due to parsing errors for all entries.
% This is a basic check; more robust error handling might be needed for specific cases.
positions = positions(any(positions,2),:);

if isempty(positions)
    fprintf('No valid position data extracted from the file.\n');
    return;
end

% --- Plot 3D Trajectory ---
fprintf('Plotting 3D trajectory...\n');
figure; % Create a new figure window
plot3(positions(:,1), positions(:,2), positions(:,3), 'LineWidth', 1.5); % Plot x, y, z
hold on; % Keep the current plot

xlabel('X Position (tx)');
ylabel('Y Position (ty)');
zlabel('Z Position (tz)');
title(['3D Trajectory from ', tumFilePath], 'Interpreter', 'none'); % 'none' interpreter for file path
grid on;
axis equal; % Use equal scaling for all axes for a more representative plot
legend('Trajectory', 'Location', 'best');
view(3); % Set 3D view
rotate3d on; % Enable interactive rotation
fprintf('Plot displayed.\n');

end