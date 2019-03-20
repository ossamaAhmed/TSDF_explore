classdef DatasetCreator
    %DATASETCREATOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        environment_params
        sdf_params
    end
    
    methods
        function obj = DatasetCreator(environment_params, sdf_params)
            % Saving the params
            obj.environment_params = environment_params;
            obj.sdf_params = sdf_params;
        end
        
        function writeNMapsToDirectory(obj, num_maps, directory, plot_flag)
            % Creating and saving a number of maps
            for map_idx = 1:num_maps
                fprintf('Creating and saving map %d/%d\n', map_idx, num_maps)

                % The environment
                environment = PolygonEnvironment(obj.environment_params);

                % Creating the SDF
                sdf_params_total = obj.sdf_params;
                sdf_params_total.bounds = environment.getBounds();
                sdf_map = SDFMap(environment, sdf_params_total);

                % Writing out
                map_idx_str = sprintf('%04d', map_idx);
                map_path = strcat(directory, '/map_', map_idx_str, '.txt');

                % Writing to file
                dlmwrite(map_path, sdf_map.values)

                % Plotting (if requested)
                if plot_flag
                    environment.plot()
                    hold on
                    sdf_map.plot()
                    hold off
                    axis equal
                    pause(0.1)
                end

            end
        end
    end
end

