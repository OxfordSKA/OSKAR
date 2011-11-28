classdef Visibilities
    %VISIBILITIES Summary of this class goes here
    %   Detailed explanation goes here
    
    % e.g. animated uu,vv plot
    %
    % for t=1:48
    %        hold on;
    %        scatter([v.uu(t), -v.uu(t)], [v.vv(t), -v.vv(t)], 'b.'); 
    %        axis square;
    %        hold off;
    %        pause(0.2);
    %        drawnow;
    % end
    %
    
    properties
        data
    end
    
    methods
        function obj = Visibilities(filename)
            obj.data = oskar_visibilities_read(filename);
        end
        
        function value = num_channels(obj)
            value = length(obj.data.frequency);
        end
        
        function value = num_times(obj)
            value = size(obj.data.uu_metres, 2);
        end
        
        function value = num_baselines(obj)
            value = size(obj.data.uu_metres, 1);
        end
        
        function value = lambda(obj, c)
            value = 299792458 / obj.data.frequency(c);
        end
        
        function value = frequency(obj, c)
            value = obj.data.frequency(c);
        end
        
        % in wavelengths. (x 2pi to put in wavenumbers for the imager)
        function values = uu(obj, t, c)
            if ~exist('c','var')
                c = 1;
            end
            values = reshape(obj.data.uu_metres(:, t, c), 1, []);
            values = values ./ obj.lambda(c);
        end
        
        function values = vv(obj, t, c)
            if ~exist('c','var')
                c = 1;
            end
            values = reshape(obj.data.vv_metres(:, t, c), 1, []);
            values = values ./ obj.lambda(c);
        end
        
        function values = ww(obj, t, c)
            if ~exist('c','var')
                c = 1;
            end
            values = reshape(obj.data.ww_metres(:, t, c), 1, []);
            values = values ./ obj.lambda(c);
        end
        
        function data = I(obj, t, c)
            if ~exist('c','var')
                c = 1;
            end
            data = reshape(obj.data.I(:, t, c), 1, []);
        end
        
        function data = Q(obj, t, c)
            if ~exist('c','var')
                c = 1;
            end
            data = reshape(obj.data.Q(:, t, c), 1, []);
        end
        
        function data = U(obj, t, c)
            if ~exist('c','var')
                c = 1;
            end
            data = reshape(obj.data.U(:, t, c), 1, []);
        end
        
        function data = V(obj, t, c)
            if ~exist('c','var')
                c = 1;
            end
            data = reshape(obj.data.V(:, t, c), 1, []);
        end
        
        
    end
    
end

