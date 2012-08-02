function plot_amp(vis, time_range, channel, pol, baseline_range, uv_plot)

if (nargin < 1)
    error('OSKAR:argChk', ...
        [...
        'Incorrect number of input arguments.\n\n' ...
        'Usage:\n'...
        '  oskar.visibilities.plot_amp(vis, <time_range>, <channel>, <pol>, <plot_3d>)\n\n' ...
        'Arguments:\n' ...
        '  1) vis (required): OSKAR MATLAB visibility structure.\n' ...
        '  2) time_range (optional): time range to plot, \n'...
        '     e.g. [1 2], default = [1 vis.num_times]\n' ...
        '  3) channel (optional): frequency channel index to plot, \n' ...
        '     default = 1\n' ...
        '  4) pol (optional): polarisation to plot (string value: xx, yy, \n'...
        '     yx, yy, I, Q, U, or V), default = ''xx''\n' ...
        '  5) baseline_range (optional): baseline index range to plot,\n' ...
        '     default = [1 vis.num_baselines]\n' ...
        '  6) uv_plot (optional): bool toggle to plot a uv plot coloured by\n'...
        '     amplitude instead of a uv distance amplitude plot, default = false\n' ...
        '\n' ...
        'Example: \n' ...
        '  oskar.visibilities.plot_amp(vis)\n' ...
        '  oskar.visibilities.plot_amp(vis, [1 2], 1, ''I'', [1 10], true)\n' ...
        ]);
end

if ~exist('time_range', 'var')
    time_range = [1 vis.num_times];
elseif (length(time_range) ~= 2)
    time_range = [time_range time_range];
end


if ~exist('channel', 'var')
    channel = 1;
end

if ~exist('pol', 'var')
    pol = 'xx';
end

if ~exist('baseline_range', 'var')
    baseline_range = [1 vis.num_baselines];
elseif (length(baseline_range) ~= 2)
    baseline_range = [baseline_range baseline_range];
end

if ~exist('uv_plot', 'var')
    uv_plot = false;
end

if (~isstruct(vis))
    error('argument vis must be an OSKAR visibilities structure');
end

if (channel > vis.num_channels)
    error('Selected channel index doesnt exist in the visibility data');
end

b = [baseline_range(1) baseline_range(2)];
t = [time_range(1) time_range(2)];
uu = vis.uu(b(1):b(2),t(1):t(2));
vv = vis.vv(b(1):b(2),t(1):t(2));
uu = [uu(:); -uu(:)];
vv = [vv(:); -vv(:)];
uu = uu./1.0e3;
vv = vv./1.0e3;
uvdist = sqrt(uu.^2 + vv.^2);

if (strcmp(pol, 'xx'))
    amp = vis.xx(b(1):b(2),t(1):t(2), channel);
elseif (strcmp(pol, 'xy'))
    amp = vis.xy(b(1):b(2),t(1):t(2), channel);
elseif (strcmp(pol, 'yx'))
    amp = vis.xy(b(1):b(2),t(1):t(2), channel);
elseif (strcmp(pol, 'yy'))
    amp = vis.xy(b(1):b(2),t(1):t(2), channel);
elseif (strcmp(pol, 'I'))
    amp = vis.I(b(1):b(2),t(1):t(2), channel);
elseif (strcmp(pol, 'Q'))
    amp = vis.Q(b(1):b(2),t(1):t(2), channel);
elseif (strcmp(pol, 'U'))
    amp = vis.U(b(1):b(2),t(1):t(2), channel);
elseif (strcmp(pol, 'V'))
    amp = vis.V(b(1):b(2),t(1):t(2), channel);
else
    error('Unknown polarisation selection');
end
amp = [amp(:); conj(amp(:))];
amp = abs(amp);

if (~uv_plot)
    scatter(uvdist, amp, 20, [0 0 1], 'filled');
    xlabel('baseline length (kilometres)');
    ylabel(['visibility (' pol ') amplitude (Jy)']);
else
    scatter(uu,vv,5,amp, 'filled')
%     uvmax = max([abs(min(uu)) abs(min(vv)) max(uu) max(vv)])
%     ti = -uvmax:0.25:uvmax;
%     [xq yq] = meshgrid(ti, ti);
%     zq = griddata(uu, vv, amp, xq, yq);
%     mesh(xq, yq, zq);
%     hold on;
%     plot3(uu, vv, amp, 'o');
%     hold off;
    
    xlabel('baseline uu (kilometres)');
    ylabel('baseline vv (kilometres)');
    colormap(jet(1024));
    c = colorbar;
    ylabel(c, ['visibility (' pol ') amplitude (Jy)'], ...
        'FontName', 'Arial', 'FontSize', 11);
end
str_title = ['UV Plot for pol: ''' pol ''' (' sprintf('%s', vis.filename) ')'];
if (time_range(1) == 0 && time_range(2) == vis.num_times)
    str_title = {str_title; ['channel ' num2str(channel) ',' num2str(length(uu(:))) ' visibilities']};
elseif (time_range(1) == time_range(2))
    str_title = {str_title; ...
        ['channel: ' num2str(channel) ', time: ' num2str(time_range(1)) ...
        ', ' num2str(length(uu(:))) ' visibilities']};
else
    str_title = {str_title; ...
        ['channel: ' num2str(channel) ', time: ' num2str(time_range(1)) ' to ' ...
        num2str(time_range(2)) ...
        ', ' num2str(length(uu(:))) ' visibilities']};
end
title(str_title, 'Interpreter', 'none');
axis square;

end
