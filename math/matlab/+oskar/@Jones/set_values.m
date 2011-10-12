function set_values( obj, values, format, location)
    %SET_VALUES Summary of this function goes here
    %   Detailed explanation goes here    
    if ~exist('location', 'var')
        location = obj.location;
    end
    obj.pointer = oskar_Jones_set_values(obj.pointer, values, format, location);
end

