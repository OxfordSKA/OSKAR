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
visM.stationIdxP = zeros(nSt,nSt);
visM.stationIdxQ = zeros(nSt,nSt);

for t=1:vis.num_times
    idx = 1;
    for j=1:vis.num_stations
        for i=(j+1):vis.num_stations
            visM.uu(i,j,t) = vis.uu(idx,t);
            visM.vv(i,j,t) = vis.vv(idx,t);
            visM.ww(i,j,t) = vis.ww(idx,t);
            idx = idx+1;
        end
    end
end

idx = 1;
for j=1:vis.num_stations
    for i=(j+1):vis.num_stations
        visM.stationIdxP(i,j) = vis.stationIdxP(idx); 
        visM.stationIdxQ(i,j) = vis.stationIdxQ(idx);
        idx = idx+1;
    end
end


if (isfield(vis, 'xx'))
    visM.xx = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'xy'))
    visM.xy = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'yx'))
    visM.yx = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'yy'))
    visM.yy = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'xx') && isfield(vis, 'xy') && isfield(vis, 'yx') && isfield(vis,'yy'))
    visM.matrix = zeros(nSt*2,nSt*2,nTi,nCh);
end

if (isfield(vis, 'I'))
    visM.I  = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'Q'))
    visM.Q  = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'U'))
    visM.U  = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'V'))
    visM.V  = zeros(nSt,nSt,nTi,nCh);
end

for c=1:vis.num_channels
    for t=1:vis.num_times
        idx = 1;
        for j=1:vis.num_stations
            for i=(j+1):vis.num_stations
                if (isfield(vis, 'xx'))
                    visM.xx(i,j,t,c) = vis.xx(idx,t,c);
                    visM.matrix((2*i)-1,(2*j)-1,t,c) = vis.xx(idx,t,c);
                end
                if (isfield(vis, 'xy'))
                    visM.xy(i,j,t,c) = vis.xy(idx,t,c);
                    visM.matrix((2*i)-1,2*j,t,c) = vis.xy(idx,t,c);
                end
                if (isfield(vis, 'yx'))
                    visM.yx(i,j,t,c) = vis.yx(idx,t,c);
                    visM.matrix(2*i,(2*j)-1,t,c) = vis.yx(idx,t,c);
                end
                if (isfield(vis, 'yy'))
                    visM.yy(i,j,t,c) = vis.yy(idx,t,c);
                    visM.matrix(2*i,2*j,t,c) = vis.yy(idx,t,c);
                end
               
                if (isfield(vis, 'I'))
                    visM.I(i,j,t,c) = vis.I(idx,t,c);
                end
                if (isfield(vis, 'Q'))
                    visM.Q(i,j,t,c) = vis.Q(idx,t,c);
                end
                if (isfield(vis, 'U'))
                    visM.U(i,j,t,c) = vis.U(idx,t,c);
                end
                if (isfield(vis, 'V'))
                    visM.V(i,j,t,c) = vis.V(idx,t,c);
                end
                
                idx = idx+1;
            end
        end
    end
end

end % End of function
