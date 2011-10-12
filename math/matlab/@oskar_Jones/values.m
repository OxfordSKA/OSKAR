function [ values ] = values(obj)
%VALUES Copies data values to matlab workspace array (matrix?)
%   Detailed explanation goes here

values = oskar_Jones_get_values(obj.pointer);

end

