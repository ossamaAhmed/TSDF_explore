classdef PolygonEnvironment < handle
    %POLYGONENVIRONMENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        polygons
        bounds
    end
    
    methods
        function obj = PolygonEnvironment(params)
            % No trajectory considered
            params.trajectory_intersection = 0;
            % Sizing the environment based on passed parameters
            % Assuming things centered around 0
            obj.bounds.max_x = params.environment_size_x / 2;
            obj.bounds.min_x = -params.environment_size_x / 2;
            obj.bounds.max_y = params.environment_size_y / 2;
            obj.bounds.min_y = -params.environment_size_y / 2;
            % Creating the environment randomly (without trajectory)
            obj.randomizeEnvironment(params);
        end
        
        % This function randomizes the environment
        function randomizeEnvironment(obj, params)
            % Creating the polygons
%             obj.polygons = cell(params.num_polygons, 1);
            polygons_temp(params.num_polygons) = Polygon();
            for polygon_index = 1:params.num_polygons
                % Randomizing the number of sides for this polygon
                polygon_pararmeters.num_sides = round((params.max_num_sides - params.min_num_sides)...
                                                * rand(1) + params.min_num_sides);
                polygon_pararmeters.size = params.polygon_size;
                % Generating a polygon
                polygon = Polygon(polygon_pararmeters);
                % Looping and finding a center location that does not intersect with other features
                place_success = 0;
                while place_success == 0; % TODO: Should put some gaurds in here against looping forever
                    % Generating the random center location
                    centroid = [  params.environment_size_x * rand(1) + obj.bounds.min_x...
                                  params.environment_size_y * rand(1) + obj.bounds.min_y ];
                    % Shift the polygon
                    polygon = polygon.setCentroid(centroid);
                    % Looping over all already placed polygons and checking for intersection
                    place_success = 1;
                    for old_polygon_index = 1:polygon_index-1
                        polygon_old = polygons_temp(old_polygon_index);
                        if polygon.intersectsWithPolygon(polygon_old)
                            place_success = 0;
                            continue
                        end
                    end
                end
                % Adding the polygon to the list
                polygons_temp(polygon_index) = polygon;
            end
            % Saving to object
            obj.polygons = polygons_temp;
        end
        
        function bounds = getBounds(obj)
            bounds = obj.bounds;
        end
        
        function num_polygons = numPolygons(obj)
            num_polygons = length(obj.polygons);
        end
        
        % Plots the enviroment
        function plot(obj, color)
            if nargin < 2
                color = '';
            end
            holdstate = ishold;
            obj.polygons(1).plot(color);
            hold on
            for polygon_index = 2:length(obj.polygons)
                obj.polygons(polygon_index).plot(color);
            end
            if ~holdstate
              hold off
            end
        end
        
    end
end

