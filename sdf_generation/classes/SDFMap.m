classdef SDFMap < handle
    %SDFMAP An SDF map.
    
    properties
        % Data
        values
        % Properties
        centers
        voxel_dim
        voxel_num
        bounds
        % Parameters
        points_per_dist
    end
    
    methods
        function obj = SDFMap(environment, params)
            % Saving the voxel dimension
            obj.voxel_dim = params.voxel_dim;
            % This is the number of points per unit distance between vertices
            % will allow for variable vector lengths, which will be more efficient!
            obj.points_per_dist = 40;
            % Setting the map parameters
            obj.setMapParameters(params.bounds);
            % Getting the values
            obj.generateDistanceFunction(environment);
        end
        
        % Sets the maps parameters based on 
        function setMapParameters(obj, bounds)
            obj.voxel_num.x = ceil((bounds.max_x - bounds.min_x) / obj.voxel_dim);
            obj.voxel_num.y = ceil((bounds.max_y - bounds.min_y) / obj.voxel_dim);
            map_size_x = obj.voxel_num.x * obj.voxel_dim;
            map_size_y = obj.voxel_num.y * obj.voxel_dim;
            map_center_x = (bounds.max_x + bounds.min_x) / 2;
            map_center_y = (bounds.max_y + bounds.min_y) / 2;
            obj.bounds.min_x = map_center_x - map_size_x/2;
            obj.bounds.min_y = map_center_y - map_size_y/2;
            obj.bounds.max_x = map_center_x + map_size_x/2;
            obj.bounds.max_y = map_center_y + map_size_y/2;
            obj.centers.x = obj.bounds.min_x+obj.voxel_dim/2:obj.voxel_dim:obj.bounds.max_x-obj.voxel_dim/2;
            obj.centers.y = obj.bounds.min_y+obj.voxel_dim/2:obj.voxel_dim:obj.bounds.max_y-obj.voxel_dim/2;
        end
        
        function generateDistanceFunction(obj, environment)
            %   Note(alexmillane): Taken from the work of Adrian Esser.
            %                      My exact implementation is available but
            %                      is apparently much slower.
            
            % Meshgrid for all X and Y values
            [X,Y] = meshgrid(obj.centers.x, obj.centers.y);
            poly_bool = ones(size(X));

            px = [];
            py = [];
            for i=1:environment.numPolygons()
                poly_cur = environment.polygons(i);
                % Compute the new set of points that are outside ALL polygons
                [not_in_flag] = ~inpolygon(X,Y,poly_cur.verticies(:,1), poly_cur.verticies(:,2));
                poly_bool = poly_bool.*not_in_flag;
                % Stack together set of points of edges
                for j=1:(size(poly_cur.verticies)-1)
                    x1 = poly_cur.verticies(j,1);
                    x2 = poly_cur.verticies(j+1,1);
                    y1 = poly_cur.verticies(j,2);
                    y2 = poly_cur.verticies(j+1,2);
                    % number of points to use for segment
                    d = floor(obj.points_per_dist*sqrt((x1-x2)^2 + (y1-y2)^2));

                    px = [px ; linspace(x1,x2,d)'];
                    py = [py ; linspace(y1,y2,d)'];
                end
                x1 = poly_cur.verticies(end,1);
                x2 = poly_cur.verticies(1,1);
                y1 = poly_cur.verticies(end,2);
                y2 = poly_cur.verticies(1,2);
                d = floor(obj.points_per_dist*sqrt((x1-x2)^2 + (y1-y2)^2));

                px = [px ; linspace(x1,x2,d)'];
                py = [py ; linspace(y1,y2,d)'];
            end

            w = length(px); % this is the number of edge points generated
            dist = [];
            for i = 1:size(X, 1) % loop over rows
                x_row = X(i,:);
                y_row = Y(i,:);

                x_tiled = repmat(x_row, 1, w);
                y_tiled = repmat(y_row, 1, w);

                px_tiled = reshape(repmat(px, 1, length(x_row))', 1, []);
                py_tiled = reshape(repmat(py, 1, length(y_row))', 1, []);

                dist_lin = sqrt( (x_tiled - px_tiled).^2 + (y_tiled - py_tiled).^2 );
                dist_row = reshape(dist_lin, length(x_row), [])';
                dist_row = min(dist_row, [], 1);

                dist = [dist ; dist_row];
            end
            % Unsigned distance
            %obj.values = dist.*poly_bool;
            % Signed distance
            obj.values = dist;
            obj.values(~poly_bool) = -1 * obj.values(~poly_bool);
        end
        
        function plot(obj)
            holdstate = ishold;
            hold on
            [X,Y] = meshgrid(obj.centers.x, obj.centers.y);
            surf(X, Y, obj.values);
            shading interp;
            view(0,90);
            if ~holdstate
              hold off
            end
        end
        
    end
end

