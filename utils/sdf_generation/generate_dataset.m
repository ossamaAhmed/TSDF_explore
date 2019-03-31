%% generate_environment (script)
%
% --
%
% (c)   March 2019 by
%       Alexander Millane
%       ETH Zurich, ASL
%       alexander.millane(at)mavt.ethz.ch
%     
%%

% Initialization
clear;
clc;
close all

%% Getting an environment

% Parameters of the training data
map_size_px = 256; % pixel size of submap
voxel_dim = 0.05; % voxel dimension

% Environment Parameters
environment_params.num_polygons = 10;
environment_params.max_num_sides = 6;
environment_params.min_num_sides = 3;
environment_params.extra_size = 1.0;
environment_params.polygon_size = 5;
environment_params.environment_size_x = map_size_px * voxel_dim;
environment_params.environment_size_y = map_size_px * voxel_dim;

% SDF Parameters
sdf_params.voxel_dim = voxel_dim;

%% Getting data from the map

% Folder to write in
output_root = '/home/ossama/Development/PLR/asldoc-2019-plr-ossama-yimeng/sdf_generation/output';
training_dir = strcat(output_root, '/training/raw');
validation_dir = strcat(output_root, '/validation/raw');

% Number of maps to generate
n_training = 10; %400;
n_validation = 2; %50;

% Object to do creation
dataset_creator = DatasetCreator(environment_params, sdf_params);

% Creating the datasets
plot_flag = true;
dataset_creator.writeNMapsToDirectory(n_training, training_dir, plot_flag);
dataset_creator.writeNMapsToDirectory(n_validation, validation_dir, plot_flag);
