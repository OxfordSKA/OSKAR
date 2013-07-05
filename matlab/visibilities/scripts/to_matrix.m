function [visM] = to_matrix(vis)

if (nargin < 1)
    error('OSKAR:argChk', ...
        [...
        '\nERROR:\n' ...
        '\tIncorrect number of input arguments.\n\n' ...
        'Usage:\n'...
        '\toskar.visibilities.to_matrix(vis)\n\n' ...
        '' ...
        'Arguments:\n' ...
        '\t1) vis (required): OSKAR MATLAB visibility structure.\n' ...
        '\n' ...
        'Example: \n' ...
        '\toskar.visibilities.to_matrix(vis)\n' ...
        ]);
end

if (~isstruct(vis))
    error('Argument vis must be an OSKAR visibilities structure.');
end


% Declare output matrix structure.
visM = vis;

nSt = vis.num_stations;
nTi = vis.num_times;
nCh = vis.num_channels;

visM.uu = zeros(nSt,nSt,nTi);
visM.vv = zeros(nSt,nSt,nTi);
visM.ww = zeros(nSt,nSt,nTi);

for t=1:vis.num_times
    idx = 1;
    for j=1:vis.num_stations
        for i=(j+1):vis.num_stations
            visM.uu(j,i,t) = vis.uu(idx,t);
            visM.vv(j,i,t) = vis.vv(idx,t);
            visM.ww(j,i,t) = vis.ww(idx,t);
            visM.uu(i,j,t) = -vis.uu(idx,t);
            visM.vv(i,j,t) = -vis.vv(idx,t);
            visM.ww(i,j,t) = -vis.ww(idx,t);
            idx = idx+1;
        end
    end
end

visM.xx = zeros(nSt,nSt,nTi,nCh);
visM.xy = zeros(nSt,nSt,nTi,nCh);
visM.yx = zeros(nSt,nSt,nTi,nCh);
visM.yy = zeros(nSt,nSt,nTi,nCh);
visM.B  = zeros(2,2,nSt,nSt,nTi,nCh);
visM.I  = zeros(nSt,nSt,nTi,nCh);
visM.Q  = zeros(nSt,nSt,nTi,nCh);
visM.U  = zeros(nSt,nSt,nTi,nCh);
visM.V  = zeros(nSt,nSt,nTi,nCh);


for c=1:vis.num_channels
    for t=1:vis.num_times
        idx = 1;
        for j=1:vis.num_stations
            for i=(j+1):vis.num_stations
                
                visM.xx(j,i,t,c) = vis.xx(idx,t,c);
                visM.xy(j,i,t,c) = vis.xy(idx,t,c);
                visM.yx(j,i,t,c) = vis.yx(idx,t,c);
                visM.yy(j,i,t,c) = vis.yy(idx,t,c);
                visM.xx(i,j,t,c) = conj(vis.xx(idx,t,c));
                visM.xy(i,j,t,c) = conj(vis.xy(idx,t,c));
                visM.yx(i,j,t,c) = conj(vis.yx(idx,t,c));
                visM.yy(i,j,t,c) = conj(vis.yy(idx,t,c));
                
                visM.B(1,1,j,i,t,c) = vis.xx(idx,t,c);
                visM.B(1,2,j,i,t,c) = vis.xy(idx,t,c);
                visM.B(2,1,j,i,t,c) = vis.yx(idx,t,c);
                visM.B(2,2,j,i,t,c) = vis.yy(idx,t,c);
                visM.B(1,1,i,j,t,c) = conj(vis.xx(idx,t,c));
                visM.B(1,2,i,j,t,c) = conj(vis.xy(idx,t,c));
                visM.B(2,1,i,j,t,c) = conj(vis.yx(idx,t,c));
                visM.B(2,2,i,j,t,c) = conj(vis.yy(idx,t,c));
                
                visM.I(j,i,t,c) = vis.I(idx,t,c);
                visM.Q(j,i,t,c) = vis.Q(idx,t,c);
                visM.U(j,i,t,c) = vis.U(idx,t,c);
                visM.V(j,i,t,c) = vis.V(idx,t,c);
                visM.I(i,j,t,c) = conj(vis.I(idx,t,c));
                visM.Q(i,j,t,c) = conj(vis.Q(idx,t,c));
                visM.U(i,j,t,c) = conj(vis.U(idx,t,c));
                visM.V(i,j,t,c) = conj(vis.V(idx,t,c));
              
                idx = idx+1;
            end
        end
    end
end

end % End of function
