function plot(Image, fov_deg, fig_title)
% PLOT Plots an image in RA, Dec.
%
% Inputs:
% Image     : Image to plot, as a square matrix.
% fov_deg   : The field of view in degrees.
% title     : The optional image title string.
% 
% Outputs:
% handle    : Handle to the figure.

if (nargin < 2)
    error('OSAKR:argChk', ...
        '\nERROR:\n\t%s.\n\nUsage:\n\t%s.%s.%s(%s)\n',...
        'Invalid arguments', 'oskar', 'image', 'plot', ...
        '<data>, <FOV in deg>, [figure title]');
end

% Set the image title, if not already set.
if ~exist('fig_title', 'var')
    fig_title = 'Image';
end

% Set up the figure.
width  = 800;
height = 600;
screen_size = get(0, 'ScreenSize');
left   = (screen_size(3) - (width  + 50));
bottom = (screen_size(4) - (height + 75));
set(gcf, 'PaperUnits', 'points');
set(gcf, 'PaperSize', [width height]);
set(gcf, 'Position', [left bottom width height]);
set(gcf, 'PaperPositionMode','auto');
set(gcf, 'PaperPosition', [left bottom width height]);
%set(gcf, 'Menubar',  'none');
%set(gcf, 'Toolbar',  'none');
set(gcf, 'Color',  'black');
set(gcf, 'InvertHardcopy','off');

axis equal;

% Plot the image.
rel_coords = linspace(-fov_deg/2, fov_deg/2, size(Image, 2));
imagesc(rel_coords, rel_coords, fliplr(Image'));

% Set axis options.
set(gca, 'YDir', 'normal');
set(gca, 'XDir', 'reverse');
set(gca, 'XColor', 'white');
set(gca, 'YColor', 'white');
set(gca, 'ZColor', 'white');
xlabel('Relative RA * cos(Dec) [deg]', 'FontSize', 11, 'FontName', 'Arial');
ylabel('Relative Dec [deg]', 'FontSize', 11, 'FontName', 'Arial');
title(fig_title, 'FontSize', 11, 'Color', 'white', 'FontName', 'Arial');
colormap(jet(1024));
c = colorbar;
% ylabel(c, 'Brightness', 'FontName', 'Arial', 'FontSize', 11);
axis square;
grid on;

end
