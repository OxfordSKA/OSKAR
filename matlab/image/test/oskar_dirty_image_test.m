u = 2.0;
v = 0.0;

uu   = [-u u];
vv   = [-v v];
amp  = [complex(1,0) complex(1,0)] ;
freq = 299792458.0;
np   = 128;
fov  = 0.5 * (180 / pi);

image = oskar_dirty_image(uu, vv, amp, freq, np, fov);

fid = figure();
oskar_plot_image(image, fov, 'dirty image test', fid);

