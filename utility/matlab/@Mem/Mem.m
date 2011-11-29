classdef Mem < handle
    %MEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(SetAccess = private, GetAccess = public, Hidden = true)
        pointer = 0;
    end
    
    methods
        function obj = Mem(type, location, num_elements, owner)

            % Default arguments.
            if ~exist('num_elements', 'var')
                num_elements = 0;
            end
            if ~exist('owner', 'var')
                owner = true;
            end
            
            % Check type and location arguments are valid.
            if (strcmp(class(type), 'oskar.Type') == 0)
                error('type argument must be an object of class oskar.Type')
            end
            if (strcmp(class(location), 'oskar.Location') == 0)
                error('location argument must be an object of class oskar.Location')
            end            
            
            % Call the mex constructor to return a pointer value.
            obj.pointer = oskar_mem_constructor(type.int32, ...
                location.int32, num_elements, owner);
        end
        
        function delete(obj)
            fprintf('delete...!\n');
            if (obj.pointer == 0)
                return;
            end
            oskar_mem_destructor(obj.pointer);
            obj.pointer = 0;
        end
    end
    
end

