% MATLAB Script to Convert TrakSTAR .mat data to TUM Trajectory Format
% (Revised based on tracker_stream_save.m analysis)

% --- Configuration ---
% >>>>> USER: SET YOUR INPUT AND OUTPUT FILE NAMES HERE <<<<<
inputMatFile = "phantom\test_left_b_short.mat"; % Replace with your Data_Sensor_X.mat file name
outputTxtFile = 'gt.txt'; % Name of the output TUM file (using a new name)

% --- CONFIRM THESE SETTINGS BASED ON YOUR SYSTEM/NEEDS ---
% 1. Euler Angle Order for TrakSTAR's A, E, R (Azimuth, Elevation, Roll)
%    'ZYX' assumes A=Yaw (around Z), E=Pitch (around new Y'), R=Roll (around new X'')
%    This is a common interpretation.
eulerAngleOrder = 'ZYX';

% 2. Angles from save_Stream_Data.m are in DEGREES.
anglesAreInDegrees = true; % This should be TRUE.

% 3. Positions from save_Stream_Data.m are in INCHES.
%    Set 'convertPositionToMeters' to true to convert to meters for TUM.
convertPositionToMeters = true;
inchToMeterConversionFactor = 0.0254; % 1 inch = 0.0254 meters

% --- Load Data ---
fprintf('Loading .mat file: %s\n', inputMatFile);
try
    loadedData = load(inputMatFile);
catch ME
    fprintf('Error loading .mat file: %s\n', ME.message);
    fprintf('Please ensure the file path is correct and the file is a valid .mat file.\n');
    return; % Exit script if loading fails
end

% The .mat file from save_Stream_Data.m should contain a variable named 'temp'
if isfield(loadedData, 'temp')
    data = loadedData.temp;
    fprintf('Successfully loaded variable "temp" from the .mat file.\n');
else
    % Fallback if 'temp' isn't found (original logic, less ideal)
    fprintf('Warning: Variable "temp" not found directly in the .mat file.\n');
    fields = fieldnames(loadedData);
    if numel(fields) == 1
        data = loadedData.(fields{1});
        fprintf('Assuming the only variable "%s" contains the data.\n', fields{1});
    else
        % Attempt to find a numeric matrix if multiple variables exist
        isNumericMatrix8Col = cellfun(@(f) isnumeric(loadedData.(f)) && ismatrix(loadedData.(f)) && size(loadedData.(f),2) == 8, fields);
        if sum(isNumericMatrix8Col) == 1
            foundFieldName = fields{isNumericMatrix8Col};
            data = loadedData.(foundFieldName);
            fprintf('Using the only numeric matrix variable "%s" with 8 columns.\n', foundFieldName);
        else
            error('Variable "temp" not found, and could not automatically identify a suitable 8-column data matrix. Please check the .mat file structure or specify the variable name.');
        end
    end
end

% Validate data structure (expecting 8 columns from 'temp')
if size(data, 2) ~= 8
    error('Data matrix must have 8 columns (pos_x, pos_y, pos_z, angle_a, angle_e, angle_r, timestamp, quality). Found %d columns.', size(data,2));
end
fprintf('Data matrix loaded successfully. Size: %d rows, %d columns.\n', size(data,1), size(data,2));

% --- Data Extraction (based on 'temp' variable structure) ---
% Col 1-3: Position (x, y, z) - in original units (inches)
% Col 4-6: Orientation angles (a, e, r) - in original units (degrees)
% Col 7: Timestamp
% Col 8: Quality (not used in TUM format, but available)

positions_original_units = data(:, 1:3);
orientations_original_units = data(:, 4:6); % Angles (a, e, r)
timestamps = data(:, 7);
% quality_data = data(:, 8); % Available if needed

% --- Data Transformation ---
fprintf('\nStarting data transformation...\n');

% 1. Convert Position Units (if specified)
positions_for_tum = positions_original_units; % Default to original
if convertPositionToMeters
    positions_for_tum = positions_original_units * inchToMeterConversionFactor;
    fprintf('Step 1: Converted positions from inches to meters.\n');
else
    fprintf('Step 1: Positions kept in original units (assumed inches).\n');
end

% 2. Convert Euler Angles to Quaternions
numRows = size(data, 1);
quaternions_for_tum = zeros(numRows, 4); % For TUM format [qx, qy, qz, qw]

orientations_for_eul2quat = orientations_original_units; % Start with original angle values

if anglesAreInDegrees
    orientations_for_eul2quat = deg2rad(orientations_original_units); % Convert to radians
    fprintf('Step 2a: Converted Euler angles from degrees to radians.\n');
else
    fprintf('Step 2a: Assuming Euler angles are already in radians (anglesAreInDegrees = false).\n');
end

fprintf('Step 2b: Converting Euler angles to quaternions using Euler sequence: %s (intrinsic).\n', eulerAngleOrder);
for i = 1:numRows
    % eul2quat expects angles in radians.
    % eul = [rotAngle1, rotAngle2, rotAngle3]
    % If sequence is 'ZYX', eul = [eulZ, eulY, eulX] (interpreted as Yaw, Pitch, Roll)
    % We assume TrakSTAR's [a, e, r] directly map to this expected order.
    current_euler_angles_rad = orientations_for_eul2quat(i,:);
    
    % MATLAB's eul2quat returns quaternion as [qw, qx, qy, qz]
    quat_matlab_order = eul2quat(current_euler_angles_rad, eulerAngleOrder);
    
    % TUM format requires quaternion as [qx, qy, qz, qw]
    quaternions_for_tum(i,:) = [quat_matlab_order(2), quat_matlab_order(3), quat_matlab_order(4), quat_matlab_order(1)];
end
fprintf('Step 2c: Successfully converted Euler angles to quaternions (TUM order: qx, qy, qz, qw).\n');

% --- Write to TUM File ---
fprintf('\nWriting data to TUM file: %s\n', outputTxtFile);
fileID = fopen(outputTxtFile, 'w');
if fileID == -1
    error('Cannot open file for writing: %s. Check permissions or path.', outputTxtFile);
end

for i = 1:numRows
    fprintf(fileID, '%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n', ...
            timestamps(i), ...
            positions_for_tum(i, 1), positions_for_tum(i, 2), positions_for_tum(i, 3), ...
            quaternions_for_tum(i, 1), quaternions_for_tum(i, 2), quaternions_for_tum(i, 3), quaternions_for_tum(i, 4));
end

fclose(fileID);
fprintf('Conversion complete. Output saved to %s\n\n', outputTxtFile);

% --- Plot 3D Trajectory (using positions as saved in the TUM file) ---
fprintf('Plotting 3D trajectory (units as saved in TUM file)...\n');
figure; % Create a new figure window
plot3(positions_for_tum(:,1), positions_for_tum(:,2), positions_for_tum(:,3), '.-'); % Plot x, y, z
hold on; % Keep the current plot
plot3(positions_for_tum(1,1), positions_for_tum(1,2), positions_for_tum(1,3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Mark start point
plot3(positions_for_tum(end,1), positions_for_tum(end,2), positions_for_tum(end,3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % Mark end point
hold off; % Release the plot

xlabel_str = 'X Position';
ylabel_str = 'Y Position';
zlabel_str = 'Z Position';
if convertPositionToMeters
    unit_suffix = ' (m)';
else
    unit_suffix = ' (inches)';
end
xlabel([xlabel_str, unit_suffix]);
ylabel([ylabel_str, unit_suffix]);
zlabel([zlabel_str, unit_suffix]);

title(['3D Trajectory (as saved in TUM file: ', outputTxtFile, ')']);
grid on;
axis equal; % Use equal scaling for all axes for a more representative plot
legend({'Trajectory', 'Start Point', 'End Point'}, 'Location', 'best');
fprintf('Plot displayed.\n');