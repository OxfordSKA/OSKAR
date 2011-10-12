function set_values( obj, values, format, location)
    %SET_VALUES Sets the values of an oskar_Jones structure from a matlab
    %array.
    %
    if ~exist('location', 'var')
        location = obj.location;
    end
    obj.pointer = oskar.Jones_set_values(obj.pointer, values, format, location);
end

