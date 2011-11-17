function [ handle ] = oskar_plot_image(Image, fov_deg, fig_title, fig_id)
% OSKAR_PLOT_IMAGE Plots an image in RA, Dec.
%
% Inputs:
% Image     : Image to plot, as a square matrix.
% fov_deg   : The field of view in degrees.
% title     : The optional image title string.
% figure_id : The optional figure ID to use.
% 
% Outputs:
% handle    : Handle to the figure.

% Set the figure ID, if not already set.
if ~exist('fig_id', 'var')
    fig_id = 991;
end

% Set the image title, if not already set.
if ~exist('fig_title', 'var')
    fig_title = 'Image';
end

if ~exist('fig_id', 'var')
    fig_id = figure();
end

% Set up the figure.
handle = figure(fig_id);
%screen_size = get(0, 'ScreenSize');
%set(gcf, 'Position', [(screen_size(3) - 700) (screen_size(4) - 700) 625 625]);
%set(gcf, 'Name', fig_title, 'NumberTitle', 'off');
set(gcf, 'Color',  'black');
% width  = 1280; % hd720
% height = 720;
% screen_size = get(0, 'ScreenSize');
% set(gcf,'InvertHardcopy','off');
% set(gcf, 'PaperUnits', 'points');
% set(gcf, 'PaperSize', [width height]);
% left   = (screen_size(3) - (width  + 50));
% bottom = (screen_size(4) - (height + 50));
% set(gcf, 'Position', [left bottom width height]);
% set(gcf,'PaperPositionMode','auto');
% set(gcf, 'PaperPosition', [left bottom width height]);
%set(gcf, 'Menubar',  'none');
%set(gcf, 'Toolbar',  'none');
axis equal;

% Plot the image.
rel_coords  = linspace(-fov_deg / 2, fov_deg / 2, size(Image, 2));
imagesc(rel_coords, -rel_coords, fliplr(Image));

% Set axis options.
set(gca, 'YDir', 'normal');
set(gca, 'XDir', 'reverse');
set(gca, 'XColor', 'white');
set(gca, 'YColor', 'white');
set(gca, 'ZColor', 'white');
xlabel('Relative RA * cos(Dec) [deg]', 'FontSize', 16);
ylabel('Relative Dec [deg]', 'FontSize', 16);
title([fig_title ...
    ' (' num2str(size(Image, 1)) 'x' num2str(size(Image, 2)) ')'], ...
    'FontSize', 18, 'Color', 'white');
c = colorbar;
colormap(jet(1024));
%caxis([-2e5 0.35*7e6]);
%caxis(caxis) freeze
%max(Image(:))
%min(Image(:))
ylabel(c, 'Brightness');
axis square;
grid on;
%set(gca, 'GridLineStyle','-');

end
