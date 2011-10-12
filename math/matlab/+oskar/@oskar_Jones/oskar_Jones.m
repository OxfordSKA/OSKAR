classdef oskar_Jones <  handle
    %OSKAR_JONES Summary of this class goes here
    %   Detailed explanation goes here
       
    % Private attributes
    properties(SetAccess = private, GetAccess = public, Hidden = true)
        pointer = 0; % pointer to oskar_Jones structure.
    end
    
    % Public methods
    methods
        % Constructor.
        %
        % FIXME: double memory type on GPU on unsupported architecture.
        %
        % NOTE! Argument order of sources and stations has to be this way
        % round to match other matlab functions
        %      e.g. ones(num_sources, num_stations)
        %
        % Usage:
        %   J = oskar_Jones(values, format, [location]);
        %   J = oskar_Jones(num_sources, num_stations, [format], [type], [location])
        %
        % Examples:
        %   J = oskar_Jones(zeros(2,2), 'scalar');
        %   J = oskar_Jones(single(zeros(2,2)), 'scalar');
        %   J = oskar_Jones(zeros(2,2,200,25), 'matrix', 'gpu');
        %   J = oskar_Jones(complex(ones(2,2,200,25), 0.5), 'matrix', 'cpu');
        %
        %   J = oskar_Jones(10, 2);
        %   J = oskar_Jones(10, 2, 'scalar', 'double', 'cpu');
        %   J = oskar_Jones(10, 2, 'matrix', 'single', 'gpu');
        %
        function obj = oskar_Jones(varargin)            
            
            default_format   = 'matrix';
            default_type     = 'double';
            default_location = 'cpu';
            
            % Switch on the number of arguments to determine which
            % version of the constructor to run.
            if (nargin == 2)
                if (ischar(varargin{2}))
                    values = varargin{1};
                    format = varargin{2};
                    if (strcmp(format, 'matrix'))
                        [~, ~, num_sources, num_stations] = size(values);
                    else
                        [num_sources, num_stations] = size(values);
                    end
                    obj.pointer = oskar_Jones_constructor(num_stations, ...
                        num_sources, format, default_type, default_location);
                    obj.set_values(values, format, default_location);
                else
                    num_sources  = varargin{1};
                    num_stations = varargin{2};
                    obj.pointer = oskar_Jones_constructor(num_stations, ...
                        num_sources, default_format, default_type, ...
                        default_location);
                    oskar_Jones_set_real_scalar(obj.pointer, 0.0);
                end                   
            
            elseif (nargin == 3)
                if (ischar(varargin{2}))
                    values   = varargin{1};
                    format   = varargin{2};
                    location = varargin{3};
                    if (strcmp(format, 'matrix'))
                        [~, ~, num_sources, num_stations] = size(values);
                    else
                        [num_sources, num_stations] = size(values);
                    end
                    obj.pointer = oskar_Jones_constructor(num_stations, ...
                        num_sources, format, default_type, location);
                    obj.set_values(values, format, location);
                else
                    num_sources  = varargin{1};
                    num_stations = varargin{2};
                    format       = varargin{3};
                    obj.pointer = oskar_Jones_constructor(num_stations, ...
                        num_sources, format, default_type, ...
                        default_location);
                    oskar_Jones_set_real_scalar(obj.pointer, 0.0);
                end
                
            elseif (nargin == 4)
                num_sources  = varargin{1};
                num_stations = varargin{2};
                format       = varargin{3};
                type         = varargin{4};
                obj.pointer = oskar_Jones_constructor(num_stations, ...
                        num_sources, format, type, default_location);
                oskar_Jones_set_real_scalar(obj.pointer, 0.0);
            
            elseif (nargin == 5)
                num_sources  = varargin{1};
                num_stations = varargin{2};
                format       = varargin{3};
                type         = varargin{4};
                location     = varargin{5};
                obj.pointer = oskar_Jones_constructor(num_stations, ...
                    num_sources, format, type, location);
                oskar_Jones_set_real_scalar(obj.pointer, 0.0);
                
            else
                error('Usage:\n %s\n %s\n', ...
                       'J = oskar_Jones(num_sources, num_stations, [format], [type], [location])', ...
                       'J = oskar_Jones(values, format, [location])');
            end           
            
        end % constructor.
         
        % Destructor
        function delete(obj)
            if (obj.pointer == 0)
                return;
            end
            oskar_Jones_destructor(obj.pointer);
            obj.pointer = 0;
        end
                
        % Copy method (defined in function file)
        obj_copy = copy(obj, location)
           
        % Return the Jones object data as a matlab array.
        data = values(obj)
        
        % returns the dimensions of the Jones array.
        function dims = size(obj)
            if strcmp(obj.format, 'scalar')
                dims = [obj.num_sources, obj.num_stations];
            else
                dims = [2, 2, obj.num_sources, obj.num_stations];
            end
        end
        
        % Set memory from matlab array object.
        %
        % For 'matrix' format the array should 4-dimensional.
        %    e.g. for ns sources and na stations:
        %      values = zeros(2,2,ns,na);
        %      values = complex(ones(2,2,ns,na), zeros(2,2,ns,na));
        %      values = single(complex(zeros(2,2), 0.5));
        %
        % For 'scalar' format the array should be 2-dimensional.
        %    e.g. for ns sources and na stations:
        %       values = ones(ns, na);
        %
        % location is optional but can be either: 'gpu' or 'cpu'
        % if not specified the location of the current memory is preserved.
        set_values(obj, values, format, location);
        
        % Set the value of the Jones matrix a real scalar.
        % For the case of matrix format the diagonal is set.
        function set_real_scalar(obj, value)
            oskar_Jones_set_real_scalar(obj.pointer, value);
        end
        
        % this = this * other 
        % e.g. J1.join_from_right(J2)
        %      J1 = J1 * J2
        % TODO: change name to multiply?!
        function join_from_right(obj, other_Jones)
            oskar_Jones_join_from_right(obj.pointer, other_Jones.pointer);
        end

        % join from left: other = other * this
        % e.g. J1.join_from_left(J2)
        %      J2 = J2 * J1
        % ...??? same as J2.join_from_right(J1) ??
        % J2 = J1 * J1
                
        % Accessor methods.
        function value = num_sources(obj)
            value = oskar_Jones_get_parameter(obj.pointer, 'num_sources');
        end
        function value = num_stations(obj)
            value = oskar_Jones_get_parameter(obj.pointer, 'num_stations');
        end
        function value = type(obj)
            value = oskar_Jones_get_parameter(obj.pointer, 'type');
        end
        function value = format(obj)
            value = oskar_Jones_get_parameter(obj.pointer, 'format');
        end
        function value = location(obj)
            value = oskar_Jones_get_parameter(obj.pointer, 'location');
        end
    end
    
    
    % Static methods.
    methods (Static = true)
        % TODO: change name to multiply?!
        function J = join(J1, J2)
            % TODO: inplace version?
            J = oskar_Jones_join(J1.pointer, J2.pointer);
        end
    end
    
    
    methods(Static = true, Hidden = true)
        % Method to return the poiter signature of an oskar_Jones object.
        % (Used by oskar_Jones mex copy function.
        function value = get_pointer(obj)
            value = obj.pointer;
        end
    end

end
