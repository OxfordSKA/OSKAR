function settings = oskar_image_settings_init(fov_deg, size, vis)

settings.fov_deg = 2;
if exist('fov_deg', 'var')
    settings.fov_deg = fov_deg;
end

settings.size = 256;
if exist('size', 'var')
    settings.size = size;
end

settings.channel_snapshots = true;
settings.channel_range(1) = -1;
settings.channel_range(2) = -1;
if exist('vis', 'var')
    settings.channel_range(1) = 0;
    settings.channel_range(2) = vis.num_channels;
end


settings.time_snapshots = true;
settings.time_range(1) = -1;
settings.time_range(2) = -1;
if exist('vis', 'var')
    settings.time_range(1) = 0;
    settings.time_range(2) = vis.num_times;
end

settings.polarisation = oskar_image_type.I;

settings.dft = true;

settings.filename = '';
settings.fits_file = '';

end