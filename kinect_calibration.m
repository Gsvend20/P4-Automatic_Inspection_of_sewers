clear
%%
squareSize = 27;
path_color = 'sewer recordings\Calibration\Color\colorboard';
path_ir = 'sewer recordings\Calibration\IR\irboard';
ext = '.png';
co_filenames = [append(path_color, '1', ext); append(path_color, '2', ext); append(path_color, '3', ext); append(path_color, '4', ext)];
ir_filenames = [append(path_ir, '1', ext); append(path_ir, '2', ext); append(path_ir, '3', ext); append(path_ir, '4', ext)];

world_points = generateCheckerboardPoints([7,10], squareSize);

% Find images
files_color = dir([path_color, '*.png']);
files_ir = dir([path_ir, '*.png']);

% Load images
for i = 1:4
    im = imread(files_color(i).name);
    img_co{i} = double(im);
end
for i = 1:4
    im = imread(files_ir(i).name);
    img_ir{i} = double(im);
end


%for i = 1:4
%    co_filenames = co_filenames + append(path_color, files_color(i).name);
%    ir_filenames = ir_filenames + append(path_ir, files_ir(i).name);
%    %[imagePoints_ir{i}, ~] = detectPatternPoints(detector, append(path_ir, files_ir(i).name), 'HighDistortion', true);
%end

%% Find image points
%[co_imagePoints, co_boardSize, co_imagesUsed] = detectCheckerboardPoints(co_filenames, 'PartialDetections', false);
%[ir_imagePoints, ir_boardSize, ir_imagesUsed] = detectCheckerboardPoints(ir_filenames, 'PartialDetections', false);

%TODO make for loop for all images
[co_imagePoints, co_boardSize, co_imagesUsed] = detectCheckerboardPoints(co_filenames(1,:));
[ir_imagePoints, ir_boardSize, ir_imagesUsed] = detectCheckerboardPoints(ir_filenames(1,:));

%% Combine points from each list
image_points = cat(4, co_imagePoints, ir_imagePoints);

%%
[stereoParams,pairsUsed,estimationErrors] = estimateCameraParameters(image_points, world_points, 'EstimateSkew', false, 'EstimateTangentialDistortion', false, 'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'mm');
figure; showReprojectionErrors(stereoParams);
figure; showExtrinsics(stereoParams);

%% Find suitable transformation
img_co = imread(co_filenames(1,:));
img_ir = imread(ir_filenames(1,:));
[movingPoints, fixedPoints] = cpselect(img_ir, img_co, 'Wait', true);

%%
tform = fitgeotrans(movingPoints, fixedPoints, 'projective');
registered = imwarp(img_ir, tform, 'OutputView', imref2d(size(img_co)));
figure('Name', 'Persp1: Projective transformation')
imshowpair(img_co, registered)