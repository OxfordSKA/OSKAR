function [visM] = to_matrix(vis)

if (nargin < 1)
    error('OSKAR:argChk', ...
        [...
        '\nERROR:\n' ...
        '\tIncorrect number of input arguments.\n\n' ...
        'Usage:\n'...
        '\toskar.vis.to_matrix(vis)\n\n' ...
        '' ...
        'Arguments:\n' ...
        '\t1) vis (required): OSKAR MATLAB visibility structure.\n' ...
        '\n' ...
        'Example: \n' ...
        '\toskar.vis.to_matrix(vis)\n' ...
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

visM.uu_metres = zeros(nSt,nSt,nTi);
visM.vv_metres = zeros(nSt,nSt,nTi);
visM.ww_metres = zeros(nSt,nSt,nTi);
visM.station_index_p = zeros(nSt,nSt);
visM.station_index_q = zeros(nSt,nSt);

for t=1:vis.num_times
    idx = 1;
    for j=1:vis.num_stations
        for i=(j+1):vis.num_stations
            visM.uu_metres(i,j,t) = vis.uu_metres(idx,t);
            visM.vv_metres(i,j,t) = vis.vv_metres(idx,t);
            visM.ww_metres(i,j,t) = vis.ww_metres(idx,t);
            idx = idx+1;
        end
    end
end

idx = 1;
for j=1:vis.num_stations
    for i=(j+1):vis.num_stations
        visM.station_index_p(i,j) = vis.station_index_p(idx); 
        visM.station_index_q(i,j) = vis.station_index_q(idx);
        idx = idx+1;
    end
end


if (isfield(vis, 'xx_Jy'))
    visM.xx_Jy = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'xy_Jy'))
    visM.xy_Jy = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'yx_Jy'))
    visM.yx_Jy = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'yy_Jy'))
    visM.yy_Jy = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'xx_Jy') && isfield(vis, 'xy_Jy') && isfield(vis, 'yx_Jy') && isfield(vis,'yy_Jy'))
    visM.matrix = zeros(nSt*2,nSt*2,nTi,nCh);
    visM.matrix_2 = zeros(2,2,nSt,nSt,nTi,nCh);
end

if (isfield(vis, 'I_Jy'))
    visM.I_Jy  = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'Q_Jy'))
    visM.Q_Jy  = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'U_Jy'))
    visM.U_Jy  = zeros(nSt,nSt,nTi,nCh);
end
if (isfield(vis, 'V_Jy'))
    visM.V_Jy  = zeros(nSt,nSt,nTi,nCh);
end

for c=1:vis.num_channels
    for t=1:vis.num_times
        idx = 1;
        for j=1:vis.num_stations
            for i=(j+1):vis.num_stations
                if (isfield(vis, 'xx_Jy'))
                    visM.xx_Jy(i,j,t,c) = vis.xx_Jy(idx,t,c);
                    visM.matrix((2*i)-1,(2*j)-1,t,c) = vis.xx_Jy(idx,t,c);
                    visM.matrix_2(1,1,i,j,t,c) = vis.xx_Jy(idx,t,c);
                end
                if (isfield(vis, 'xy_Jy'))
                    visM.xy_Jy(i,j,t,c) = vis.xy_Jy(idx,t,c);
                    visM.matrix((2*i)-1,2*j,t,c) = vis.xy_Jy(idx,t,c);
                    visM.matrix_2(1,2,i,j,t,c) = vis.xy_Jy(idx,t,c);
                end
                if (isfield(vis, 'yx_Jy'))
                    visM.yx_Jy(i,j,t,c) = vis.yx_Jy(idx,t,c);
                    visM.matrix(2*i,(2*j)-1,t,c) = vis.yx_Jy(idx,t,c);
                    visM.matrix_2(2,1,i,j,t,c) = vis.yx_Jy(idx,t,c);
                end
                if (isfield(vis, 'yy_Jy'))
                    visM.yy_Jy(i,j,t,c) = vis.yy_Jy(idx,t,c);
                    visM.matrix(2*i,2*j,t,c) = vis.yy_Jy(idx,t,c);
                    visM.matrix_2(2,2,i,j,t,c) = vis.yy_Jy(idx,t,c);
                end
               
                if (isfield(vis, 'I_Jy'))
                    visM.I_Jy(i,j,t,c) = vis.I_Jy(idx,t,c);
                end
                if (isfield(vis, 'Q_Jy'))
                    visM.Q_Jy(i,j,t,c) = vis.Q_Jy(idx,t,c);
                end
                if (isfield(vis, 'U_Jy'))
                    visM.U_Jy(i,j,t,c) = vis.U_Jy(idx,t,c);
                end
                if (isfield(vis, 'V_Jy'))
                    visM.V_Jy(i,j,t,c) = vis.V_Jy(idx,t,c);
                end
                
                idx = idx+1;
            end
        end
    end
end

end % End of function
