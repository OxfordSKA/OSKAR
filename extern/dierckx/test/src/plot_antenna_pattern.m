close all;

% Load the data.
data = dlmread('Port1_300.txt', '', 2, 0);
data = [data; data(1:37,:)];
data(end-36:end,2) = 360;

log_scale = true;

% Get the coordinates.
theta = data(:, 1) * pi/180;
phi   = data(:, 2) * pi/180;

% Convert to complex numbers.
amp_theta = data(:, 4);
amp_phi = data(:, 6);
if (log_scale)
    amp_theta = 10.^(amp_theta/10);    
    amp_phi = 10.^(amp_phi/10);
end
% amp_theta = ones(size(amp_theta));
% amp_phi = ones(size(amp_phi));
g_theta = amp_theta .* exp(1i * data(:, 5) * pi/180);
g_phi = amp_phi .* exp(1i * data(:, 7) * pi/180);
min_db = 20;
phi = phi(theta < pi/2);
g_theta = g_theta(theta < pi/2);
g_phi = g_phi(theta < pi/2);
theta = theta(find(theta < pi/2)); %#ok<FNDSB>




phi1 = reshape(phi, 18, []);
theta1 = reshape(pi/2 - theta, 18, []);
amp1 = reshape(abs(g_theta), 18, []);
amp2 =  reshape(abs(g_phi), 18, []);
figure()
surface(phi1, theta1, amp1);
figure()
surface(phi1, theta1, amp2);


figure()
% Plot 3D (Pol 1)
% ---------------
%r1 = min_db + log10(abs(g_theta));
r1 = abs(g_theta);
r1(r1 < 0) = 0;
[x1 y1 z1] = sph2cart(phi, pi/2 - theta, r1);
x1 = reshape(x1, 18, []);
y1 = reshape(y1, 18, []);
z1 = reshape(z1, 18, []);
c1 = reshape(r1, 18, []);
surface(x1,y1,z1,c1);
%scatter3(x1(:), y1(:), z1(:), 10, r1(:));

hold on;

% Plot 3D (Pol 2)
% ---------------
% r2 = min_db + log10(abs(g_phi));
r2 = abs(g_phi);
r2(r2 < 0) = 0;


[x2 y2 z2] = sph2cart(phi, pi/2 - theta, r2);
x2 = reshape(x2, 18, []);
y2 = reshape(y2, 18, []);
z2 = reshape(z2, 18, []);
c2 = reshape(r2, 18, []);
%surface(x2,y2,z2,c2);
%scatter3(x2, y2,z2, c2);

% Plot 3D options.
%axis square
axis equal
%axis vis3d
%axis off
%shading interp
lighting phong
camlight(-60,40)
grid on;


% Plot slice
figure()
slice = 1;
plot([flipud(c1(:,slice)); c1(:,slice + 180/5)],'.-')

