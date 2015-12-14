function [ vis ] = create_template_struct(linear, all_fields, num_channels, ...
    num_times, num_stations)
%CREATE_TEMPLATE_STRUCT Creates a template visibility structure.
%   Creates a template visibility structure.
%
% Arguments
% - Linear is a bool switch to instruct the function to create a structure
%   with linear polarisation amplitude fields or Stokes-I amplitude (if 
%   set to false)
% - all_fields is a bool switch to instruct the function to create only a
%   minimal set of fields or all possible fields.
% - num_channels 
% - num_times
% - num_stations
%

if (nargin > 5)
    error('oops')
end

if ~exist('linear', 'var')
    linear = 1;
end
if ~exist('all_fields', 'var')
    all_fields = 0;
end
if ~exist('num_channels', 'var')
    num_channels = 1;
end
if ~exist('num_times', 'var')
    num_times = 1;
end
if ~exist('num_stations', 'var')
    num_stations = 2;
end

num_baselines = (num_stations*(num_stations-1))/2;

vis = struct;
vis.freq_start_hz = 0;
vis.uu_metres = zeros(num_baselines, num_times);
vis.vv_metres = zeros(num_baselines, num_times);
if (linear == 1)
    vis.xx_Jy = zeros(num_baselines, num_times, num_channels);
    vis.xy_Jy = zeros(num_baselines, num_times, num_channels);
    vis.yx_Jy = zeros(num_baselines, num_times, num_channels);
    vis.yy_Jy = zeros(num_baselines, num_times, num_channels);
else
    vis.I_Jy = zeros(num_baselines, num_times, num_channels);
end

% Write dimensions if not 1D or writing all fields.
if (num_channels > 1 || num_times > 1 || all_fields == 1)
    vis.num_channels = num_channels;
    vis.num_times = num_times;
    vis.num_baselines = num_baselines;
end


if (all_fields == 1)
    vis.num_stations = num_stations;
    vis.freq_inc_hz = 0;
    vis.channel_bandwidth_hz = 0;
    vis.time_start_mjd_utc = 0;
    vis.time_inc_seconds = 0;
    vis.time_int_seconds = 0;
    vis.phase_centre_ra_deg = 0;
    vis.phase_centre_dec_deg = 0;
    vis.telescope_lon_deg = 0;
    vis.telescope_lat_deg = 0;
    
    vis.ww_metres = zeros(num_channels, num_times, num_stations);

    vis.station_lon_deg = 0;
    vis.station_lat_deg = 0;
    
    vis.station_x_metres = zeros(num_stations);
    vis.station_y_metres = zeros(num_stations);
    vis.station_z_metres = zeros(num_stations);
    
    vis.station_orientation_x_deg = zeros(num_stations);
    vis.station_orientation_y_deg = zeros(num_stations);   
end


end

