clear all
clc
close all
format longg

image_dir = 'D:\leo_tracking\data\2020-02-23\processed\sat2\c';
% image_dir = 'C:\RealTimeChipDetection\test_chips\telkom1_s';

image_dirs = subdir(image_dir);

if strcmp(image_dirs, '')
    image_dirs = {image_dir};
end

IPS = 'D:\AnomalyDetection\ips_win64_debug.exe';
global_opt = '-write -mcc_debug_level 99 -ab -ch -em_multitarg -force_ct0 -thresh_spike 0.0';
ofp_opt = '-wl 4000 6000 -dc 0.0 -blur 1.5 -blur_det 1.5 -streak_axis_ratio 0.0';

for i = 1:length(image_dirs)
    
    dir_images = dir(image_dirs{i});
    dir_images = dir_images(3:end);
    vidx = ~cellfun(@isempty, strfind({dir_images.name}, '.jpg'));
    dir_images = dir_images(vidx);
    dir_images = dir_images(2:end);
%     dir_images = dir_images(30);
    
    out_array = zeros(6000*4000, length(dir_images));
    xflip = true;
    yflip = true;
    
    for j = 1:length(dir_images)
        disp(['Loading Image ' num2str(j)]);
        fname = [dir_images(j).folder filesep dir_images(j).name];
        image_data = imread(fname);
%         gray = 0.2989 * image_data(:,:,1) + 0.5870 ...
%                       * image_data(:,:,2) + 0.1140 ...
%                       * image_data(:,:,3);
%         if ( xflip ), gray = fliplr(gray); end
%         if ( yflip ), gray = flipud(gray); end
        out_array(:, j) = reshape(image_data, 6000*4000, 1);
    end
    
    sat_pair = strsplit(image_dirs{i}, '\');
    output_file = [image_dirs{i} filesep sat_pair{end} '.ofp'];
    
    disp('Writing to OFP');
    
    f = fopen(output_file, 'wb');
    fwrite(f, out_array, 'double');
    fwrite(f, 4000, 'int');
    fwrite(f, 6000, 'int');
    fwrite(f, length(dir_images), 'int');
    fclose(f);
    
    disp(['Running IPS for ' output_file]);
    
    [x, y] = system( [IPS ' ' global_opt ' ' output_file ' ' ofp_opt]);
    
%     ipsDir = ['C:\RealTimeChipDetection\test_chips\telkom1' filesep sat_pair{end}];
    ipsDir = [' D:\leo_tracking\chips_raw\2020-02-23' filesep sat_pair{end}];
    
    movefile([image_dirs{i} filesep '*.ips.ir3*'], ipsDir);
    movefile([image_dirs{i} filesep '*.ofp'], ipsDir);
    
end