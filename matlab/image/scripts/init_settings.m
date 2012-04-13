function settings = init_settings(fov_deg, size, vis)

settings.fov_deg = 2;
if exist('fov_deg', 'var')
    settings.fov_deg = fov_deg;
end

settings.size = 256;
if exist('size', 'var')
    settings.size = size;
end

settings.channel_snapshots = true;
settings.channel_range(1) = 0;
settings.channel_range(2) = -1;
if exist('vis', 'var')
    settings.channel_range(1) = 0;
    settings.channel_range(2) = vis.num_channels-1;
end


settings.time_snapshots = false;
settings.time_range(1) = 0;
settings.time_range(2) = -1;
if exist('vis', 'var')
    settings.time_range(1) = 0;
    settings.time_range(2) = vis.num_times-1;
end

settings.polarisation = oskar.image.type.I;

settings.transform_type = oskar.image.transform.dft_2d;

settings.filename = '';
settings.fits_file = '';

end