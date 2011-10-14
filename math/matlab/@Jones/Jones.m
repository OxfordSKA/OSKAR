classdef Jones <  handle
    %JONES Class holding a matrix of Jones matrices corresponding to a
    %number of sources and stations.
    %
    % This is a matlab handle class used as an interface to the oskar_Jones
    % structure defined in oskar/src/math/oskar_Jones.h
    %
    
    
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
        %   J = Jones(values, format, [location]);
        %   J = Jones(num_sources, num_stations, [format], [type], [location])
        %
        % Examples:
        %   J = Jones(zeros(2,2), 'scalar');
        %   J = Jones(single(zeros(2,2)), 'scalar');
        %   J = Jones(zeros(2,2,200,25), 'matrix', 'gpu');
        %   J = Jones(complex(ones(2,2,200,25), 0.5), 'matrix', 'cpu');
        %
        %   J = Jones(10, 2);
        %   J = Jones(10, 2, 'scalar', 'double', 'cpu');
        %   J = Jones(10, 2, 'matrix', 'single', 'gpu');
        %
        function obj = Jones(varargin)
            
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
                    obj.pointer = Jones_constructor(num_stations, ...
                        num_sources, format, class(values), default_location);
                    obj.set_values(values, format, default_location);
                else
                    num_sources  = varargin{1};
                    num_stations = varargin{2};
                    obj.pointer = Jones_constructor(num_stations, ...
                        num_sources, default_format, default_type, ...
                        default_location);
                    Jones_set_real_scalar(obj.pointer, 0.0);
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
                    obj.pointer = Jones_constructor(num_stations, ...
                        num_sources, format, class(values), location);
                    obj.set_values(values, format, location);
                else
                    num_sources  = varargin{1};
                    num_stations = varargin{2};
                    format       = varargin{3};
                    obj.pointer = Jones_constructor(num_stations, ...
                        num_sources, format, default_type, ...
                        default_location);
                    Jones_set_real_scalar(obj.pointer, 0.0);
                end
                
            elseif (nargin == 4)
                num_sources  = varargin{1};
                num_stations = varargin{2};
                format       = varargin{3};
                type         = varargin{4};
                obj.pointer = Jones_constructor(num_stations, ...
                    num_sources, format, type, default_location);
                Jones_set_real_scalar(obj.pointer, 0.0);
                
            elseif (nargin == 5)
                num_sources  = varargin{1};
                num_stations = varargin{2};
                format       = varargin{3};
                type         = varargin{4};
                location     = varargin{5};
                obj.pointer = Jones_constructor(num_stations, ...
                    num_sources, format, type, location);
                Jones_set_real_scalar(obj.pointer, 0.0);
                
            else
                error('Usage:\n %s\n %s\n', ...
                    'J = Jones(num_sources, num_stations, [format], [type], [location])', ...
                    'J = Jones(values, format, [location])');
            end
            
        end % constructor.
        
        % Destructor
        function delete(obj)
            if (obj.pointer == 0)
                return;
            end
            Jones_destructor(obj.pointer);
            obj.pointer = 0;
        end
        
               
        % Copy method (defined in function file)
        obj = copy(obj, location)
        
        % Return the Jones object data as a matlab array.
        data = values(obj)
        
        % returns the dimensions of the Jones array.
        function d = size(obj, dim)
            if strcmp(obj.format, 'scalar')
                dimensions = [obj.num_sources, obj.num_stations];
            else
                dimensions = [2, 2, obj.num_sources, obj.num_stations];
            end
            
            if ~exist('dim', 'var')
                d = dimensions;
            elseif (dim > ndims(dimensions))
                d = 1;
            else
                d = dimensions(dim);
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
            Jones_set_real_scalar(obj.pointer, value);
        end
        
        % this = this * other
        % e.g. J1.join_from_right(J2)
        %      J1 = J1 * J2
        % TODO: change name to multiply?!
        function join_from_right(obj, other_Jones)
            Jones_join_from_right(obj.pointer, other_Jones.pointer);
        end
              
        % Accessor methods.
        function value = num_sources(obj)
            value = Jones_get_parameter(obj.pointer, 'num_sources');
        end
        function value = num_stations(obj)
            value = Jones_get_parameter(obj.pointer, 'num_stations');
        end
        function value = type(obj)
            value = Jones_get_parameter(obj.pointer, 'type');
        end
        function value = format(obj)
            value = Jones_get_parameter(obj.pointer, 'format');
        end
        function value = location(obj)
            value = Jones_get_parameter(obj.pointer, 'location');
        end

        % overloaded matrix multiply operator
        % eg.
        % J = K * E
        % J = K * E * P
        function j3 = mtimes(j1, j2)
            j3 = Jones_join(j1.pointer, j2.pointer);
        end
        
        % Disable undefined operators
        function eq(~, ~)
            error('Equality operator (=) undefined for oskar.Jones objects');
        end
        function ne(~, ~)
            error('Not equal to (~=) operator undefined for oskar.Jones objects');
        end
        function le(~, ~)
            error('Less than or equal to (<=) operator undefined for oskar.Jones objects');
        end
        function lt(~, ~)
            error('Less than (<) operator undefined for oskar.Jones objects');
        end
        function ge(~, ~)
            error('Greater than or equal to (>=) operator undefined for oskar.Jones objects');
        end
        function gt(~, ~)
            error('Greater than operator (>) undefined for oskar.Jones objects');
        end
    end
    
       
    % Static methods.
    methods (Static = true)
        % TODO: change name to multiply?!
        function J = join(J1, J2)
            % TODO: inplace version?
            J = Jones_join(J1.pointer, J2.pointer);
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
