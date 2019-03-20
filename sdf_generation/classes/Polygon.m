classdef Polygon < handle & matlab.mixin.Copyable
    %POLYGON Encapsulates a polygon and related functionality
    %   Detailed explanation goes here
    
    properties
        relative_verticies
        verticies
        centroid
        num_sides
        side_vectors
    end
    
    methods
        
        function obj = Polygon(params)
            if nargin > 0
                % Initializes the polygon with random obj.verticies with a centroid at [0, 0]
                conv_vertices = obj.getRandomVertices(params);
                obj.setUpPolygon(conv_vertices);
            end
        end
        
        function setUpPolygon(obj, raw_vertices)
            obj.num_sides = size(raw_vertices,1);
            % Finding the centroid
            raw_centroid = obj.calculateCentroid(raw_vertices(:,1), raw_vertices(:,2));
            % Shifting the obj.verticies
            obj.relative_verticies = raw_vertices - repmat(raw_centroid, obj.num_sides,1);
            obj.centroid = [0 0];
            % Calculating the absolute obj.verticies
            obj.calculateVerticies();
            % Calculating the vectors representing the sides
            obj.side_vectors = obj.calculateSideVectors();
        end
        
        function random_vertices = getRandomVertices(obj, params)
            % Extracting the desired number of sides
            desired_num_sides = params.num_sides;
            % For now this constuctor always creates a random polygon
            % Creating initial polygon (potentially not enough sides)
            potential_verticies = rand(desired_num_sides, 2);
            convex_hull_indicies = convhull(potential_verticies(:,1), potential_verticies(:,2));
            num_sides = size(convex_hull_indicies, 1) - 1;
            % Creating random obj.verticies until we have a hull of the desired side number
            while num_sides < desired_num_sides
                % Finding the first interior point
                interior_indicies = setdiff([1:desired_num_sides], convex_hull_indicies);
                interior_index = interior_indicies(1);
                % Generating a new potential vertex
                new_potential_vertex = params.size * rand(1, 2);
                % Adding it to the potential vertex list
                potential_verticies(interior_index, :) = new_potential_vertex;
                % Recalculating the hull and the number of sides
                convex_hull_indicies = convhull(potential_verticies(:,1), potential_verticies(:,2));
                num_sides = size(convex_hull_indicies, 1) - 1;
            end
            % Reordering the obj.verticies and storing
            random_vertices = potential_verticies(convex_hull_indicies(2:end), :);
        end
                
        % Returns the obj.verticies
        function calculateVerticies(obj)
            obj.verticies = obj.relative_verticies + repmat(obj.centroid, obj.num_sides, 1);
        end
        
        % Sets the relative verticies and maintains consistency
        function setRelativeVerticies(obj, new_relative_verticies)
            % Just ensuring that these are indeed relative verticies
            new_centroid = obj.calculateCentroid(new_relative_verticies(:,1), new_relative_verticies(:,2));
            obj.relative_verticies = new_relative_verticies - repmat(new_centroid, obj.getNumSides(), 1);
            % Recalculating the stuff that depends on the verticies
            obj.calculateVerticies();
            % Calculating the vectors representing the sides
            obj.side_vectors = obj.calculateSideVectors();            
        end
        
        % Returns the obj.verticies
        function verticies = getVerticies(obj)
            verticies = obj.verticies;
        end
        
        % Returns the relative Verticies
        function relative_verticies = getRelativeVerticies(obj)
            relative_verticies = obj.relative_verticies;
        end
        
        % Returns the number of sides
        function num_sides = getNumSides(obj)
            num_sides = obj.num_sides;
        end

        % Returns the polygon centroid
        function centroid = getCentroid(obj)
            centroid = obj.centroid;
        end
        
        % Shifts the polygon so that the centroid lies where specified
        function obj = setCentroid(obj, centroid)
            obj.centroid = centroid;
            obj.calculateVerticies();
        end
        
        % Calculating and storing the side vectors
        function side_vectors = calculateSideVectors(obj)
            % The side indicies
            indicies = mod([[0:obj.num_sides-1]' [1:obj.num_sides]'], obj.num_sides) + 1;
            % Creating the vectors
            side_vectors = obj.verticies(indicies(:,2),:) - obj.verticies(indicies(:,1),:);
        end
    
        % Returns the side vectors
        function side_vectors = getSideVectors(obj)
            side_vectors = obj.side_vectors;
        end
        
        % Returns the test vectors
        function test_vectors = getTestVectors(obj, point)
            % Putting the vector relative to the obj.verticies
            test_vectors = repmat(point,obj.num_sides,1) - obj.verticies;
        end
        
        % Gets a scaled version of this polygon
        function scaled_copy = getScaledCopy(obj, scale)
            % Getting a copy
            scaled_copy = obj.copy();
            % Scaling the vertices
            scaled_relative_verticies = scale * obj.relative_verticies;
            % Setting the relative verticies in the scaled polygon
            scaled_copy.setRelativeVerticies(scaled_relative_verticies);
        end
        
        % Checks whether this polygon intersects with the passed polygon
        % - http://stackoverflow.com/questions/753140/how-do-i-determine-if-two-convex-polygons-intersect
        function intersection = intersectsWithPolygon(obj, polygon)
            % Constants
            num_polygons = 2;
            % Putting polygons in cell array
            polygons = cell(num_polygons,1);
            polygons{1} = obj;
            polygons{2} = polygon;
            % Looping over all sides in both polygons
            seperator_found = 0;
            for polygon_index = 1:num_polygons
                % Extracting a polygon
                polygon = polygons{polygon_index};
                polygon_verticies = polygon.getVerticies();
                % Looping over the sides in this polygon
                for side_index = 1:polygon.getNumSides()
                    % Extracting a side
                    start_index = side_index;
                    end_index = mod(side_index, polygon.getNumSides()) + 1;
                    side = polygon_verticies([start_index end_index],:);
                    % Testing if the side is seperating
                    seperation = polygons{1}.seperatedByVector(polygons{2}, side);
                    % Breaking if seperator found
                    if seperation == 1
                        seperator_found = 1;
                    end
                end
            end
            % Output
            intersection = ~seperator_found;
        end
        
        % Checks whether the passes vector seperates this polygon and the
        % passed polygon
        function seperation = seperatedByVector(obj, polygon, vector)
            % Constants
            num_polygons = 2;
            % Putting polygons in cell array
            polygons = cell(num_polygons,1);
            polygons{1} = obj;
            polygons{2} = polygon;
            % Shifting vector to origin
            vector = vector(1,:) - vector(2,:);
            % Creating the normal with unit norm
            vector_normal = [-vector(2) vector(1)];
            vector_normal = vector_normal./norm(vector_normal,2);
            % Projecting the points to this line
            projections = cell(num_polygons, 1);
            for polygon_index = 1:num_polygons
                % Extracting polygon
                polygon = polygons{polygon_index};
                polygon_verticies = polygon.getVerticies();
                % Looping over the obj.verticies
                num_verticies = size(polygon_verticies, 1);
                polygon_projections = zeros(num_verticies, 1);
                for vertex_index = 1:num_verticies
                    % Extracting vertex
                    vertex = polygon_verticies(vertex_index, :);
                    % Projecting the vertex
                    polygon_projections(vertex_index) = dot(vertex, vector_normal);
                end
                % Saving projections
                projections{polygon_index} = polygon_projections;
            end
            % Testing for intersection
            projections_1 = projections{1};
            projections_2 = projections{2};
            size_projections_1 = size(projections_1, 1);
            size_projections_2 = size(projections_2, 1);
            comparison_matrix = zeros(size_projections_1, size_projections_2);
            for i = 1:size_projections_2
                comparison_matrix(:,i) = projections_1 > repmat(projections_2(i), size_projections_1, 1);
            end
            seperation = ~max(max(range(comparison_matrix,1)), max(range(comparison_matrix,2)));
        end
        
        % Checks whether the passes vector seperates this polygon and the
        % passed polygon
        function distance = distanceFromPoint(obj, point)
            % Negative factor for inside points
            if obj.isPointInside(point)
                inside_factor = -1.0;
            else
                inside_factor = 1.0;
            end
            % Creating the vectors
            test_vectors = obj.getTestVectors(point);
            % Projecting test point onto side
            side_vectors_length_2 = dot(obj.side_vectors',obj.side_vectors');
            projection_distances_along_side = dot(test_vectors', obj.side_vectors') ./ side_vectors_length_2;
            projection_distances_along_side = min(max(projection_distances_along_side,0), 1);
            projection_points_on_side = [projection_distances_along_side' projection_distances_along_side'] .* obj.side_vectors;
            % Getting the distances
            distance_vecs = projection_points_on_side - test_vectors;
            distances_2 = dot(distance_vecs', distance_vecs');
            % Minimum distance
            [~,i] = min(abs(distances_2));
            distance = inside_factor * sqrt(distances_2(i));
        end
        
        % Checks whether the passes vector seperates this polygon and the
        % passed polygon
        function distance = absDistanceFromPoint(obj, point)
            % Creating the vectors
            test_vectors = obj.getTestVectors(point);
            % Projecting test point onto side
            side_vectors_length_2 = dot(obj.side_vectors',obj.side_vectors');
            projection_distances_along_side = dot(test_vectors', obj.side_vectors') ./ side_vectors_length_2;
            projection_distances_along_side = min(max(projection_distances_along_side,0), 1);
            projection_points_on_side = [projection_distances_along_side' projection_distances_along_side'] .* obj.side_vectors;
            % Getting the distances
            distance_vecs = projection_points_on_side - test_vectors;
            distances_2 = dot(distance_vecs', distance_vecs');
            % Minimum distance
            [~,i] = min(abs(distances_2));
            distance = sqrt(distances_2(i));
        end
        
        % Returns true if point is inside polygon
        function inside = isPointInside(obj, point)
            test_vectors = obj.getTestVectors(point);
            cross_product_signs = sign(Polygon.twoDCrosses(obj.side_vectors, test_vectors));
            inside = all(cross_product_signs == cross_product_signs(1));
        end
        
        % Returns the polygon bounding box
        function bounds = getBounds(obj)
            bounds.max_x = max(obj.verticies(:,1));
            bounds.min_x = min(obj.verticies(:,1));
            bounds.max_y = max(obj.verticies(:,2));
            bounds.min_y = min(obj.verticies(:,2));
        end
        
        function plot(obj, color)
            if nargin < 2
                color = '';
            end
            plot(obj.verticies([1:end 1],1), obj.verticies([1:end 1],2),color,'LineWidth',2.0)
        end
        
    end
    

    methods(Static)
        
        % The 2D cross product of two single vectors
        function product = twoDCross(vec1, vec2)
            product = vec1(1)*vec2(2) - vec1(2)*vec2(1);
        end
        
        % The 2D cross product of many vectors. Vectorized
        function product = twoDCrosses(vec1, vec2)
            vec2_perp = [vec2(:,2), -vec2(:,1)];
            product = dot(vec1', vec2_perp')';
        end
        
        
        % Gets the centroid of a polygon
        % Note(alexmillane): Taken from polygeom
        % (https://ch.mathworks.com/matlabcentral/fileexchange/319-polygeom-m)
        function centroid = calculateCentroid(x, y)
            % check if inputs are same size
            if ~isequal( size(x), size(y) )
                error( 'X and Y must be the same size');
            end
            % size
            n = length(x);
            % temporarily shift data to mean of vertices for improved accuracy
            xm = mean(x);
            ym = mean(y);
            x = x - xm*ones(n,1);
            y = y - ym*ones(n,1);
            % delta x and delta y
            dx = x( [ 2:n 1 ] ) - x;
            dy = y( [ 2:n 1 ] ) - y;
            % summations for CW boundary integrals
            A = sum( y.*dx - x.*dy )/2;
            Axc = sum( 6*x.*y.*dx -3*x.*x.*dy +3*y.*dx.*dx +dx.*dx.*dy )/12;
            Ayc = sum( 3*y.*y.*dx -6*x.*y.*dy -3*x.*dy.*dy -dx.*dy.*dy )/12;
            % check for CCW versus CW boundary
            if A < 0
                A = -A;
                Axc = -Axc;
                Ayc = -Ayc;
            end
            % centroidal moments
            xc = Axc / A;
            yc = Ayc / A;
            % replace mean of vertices
            x_cen = xc + xm;
            y_cen = yc + ym;
            % return
            centroid = [x_cen y_cen];
        end
        
    end
    
end

