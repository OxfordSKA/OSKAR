function generate_data(filename)

% Open file.
fid = fopen(filename, 'w');

% Test data.
% data = [ 1 2 4 1 1 1 1 1; ...
%     6 3 5 2 1 1 1 1; ...
%     4 2 1 5 1 1 1 1; ...
%     5 4 2 3 1 1 1 1];
% imagesc(data);
% colorbar;
% colormap(jet(10240));
% axis square; axis tight;

% Test sine wave.
% n = 10;
% x = linspace(-3.14, 3.14, n);
% y = linspace(-3.14, 3.14, n);
% data = zeros(n, n, 'single');
% for i = 1:n
%     for j = 1:n
%         data(i, j)= sin(x(i)) * sin(y(j));
%     end
% end
% imagesc(data)
% colorbar;
% colormap(jet(10240));
% axis square; axis tight;

% Test 2D Gaussians.
n = 25;
x = linspace(-4, 4, n);
y = linspace(-4, 4, n);
sigma_x_1 = 1.0;
sigma_y_1 = 0.5;
sigma_x_2 = 1.5;
sigma_y_2 = 0.3;
x_0 = 1.5;
y_0 = -2;
data = zeros(n, n, 'double');
RandStream.setDefaultStream (RandStream('mt19937ar','seed', 7));
for i = 1:n
    for j = 1:n
        data(j, i) = exp(-(x(i)^2)/(2*sigma_x_1^2) - (y(j)^2)/(2*sigma_y_1^2));
        data(j, i) = data(j, i) + 5e-4 * randn(1); % Add some Gaussian noise.
        
        data(j, i) = data(j, i) + ... % Add another Gaussian.
            0.4 * exp(-((x(i) - x_0)^2)/(2*sigma_x_2^2) - ((y(j) - y_0)^2)/(2*sigma_y_2^2));
    end
end
surf(data);
colorbar;
colormap(jet(10240));
axis square; axis tight;

% Write header.
header = zeros(1, 8, 'int32');
header(1) = size(data, 2); % size_x (fastest varying).
header(2) = size(data, 1); % size_y (slowest varying).
header(3) = sizeof('single'); % number of bytes per element.
fwrite(fid, header, 'int32');

% Write data in native/Fortran order.
fwrite(fid, data, 'single');

% Close file.
fclose(fid);

end