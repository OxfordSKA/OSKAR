function [ values ] = values(obj)
%VALUES Copies data values contained in an oskar_Jones structure to the
%MATLAB workspace.
%
% The dimensionality of the values array will depend on the format of the 
% Jones data and the number of stations and sources for which Jones
% matrices are stored.

values = oskar.Jones_get_values(obj.pointer);

end

