classdef SkyModel < handle
    %SKYMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(SetAccess = private, GetAccess = public, Hidden = true)
        pointer = 0; % pointer to oskar_SkyModel structure.
    end
    
    % Public methods
    methods
        % Constructor
        function obj = SkyModel(varargin)
            nargin
            % if .. call various constructor mex functions.
        end
        
        % destructor
        function delete(obj)
            if (obj.pointer == 0)
                return;
            end
            % SkyModel_destructor(obj.pointer)
            obj.pointer = 0;
        end
        
        % Copy method
        function copy(obj, location)
            % copy constructor equivalent
        end
        
        % Return the values as a matlab array
        function data = values(obj)
            % TODO
        end
        
        % Accessor methods
        function value = num_sources(obj)
        end                
        function value = location(obj)
        end        
        function value = type(obj)
        end
        
        
        % Disable undefined operators
        function mtimes(~, ~)
            error('Multiply operator (*) undefined for oskar.SkyModel objects');
        end
        function eq(~, ~)
            error('Equality operator (=) undefined for oskar.SkyModel objects');
        end
        function ne(~, ~)
            error('Not equal to (~=) operator undefined for oskar.SkyModel objects');
        end
        function le(~, ~)
            error('Less than or equal to (<=) operator undefined for oskar.SkyModel objects');
        end
        function lt(~, ~)
            error('Less than (<) operator undefined for oskar.SkyModel objects');
        end
        function ge(~, ~)
            error('Greater than or equal to (>=) operator undefined for oskar.SkyModel objects');
        end
        function gt(~, ~)
            error('Greater than operator (>) undefined for oskar.SkyModel objects');
        end       
    end
    
    methods(Static = true, Hidden = true)
        % Method to return the poiter signature of an oskar.SkyModel object.
        % (used by copy function)
        function value = get_pointer(obj)
            value = obj.pointer;
        end
    end
    
end

