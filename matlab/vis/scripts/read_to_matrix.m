function visM = read_to_matrix(filename)
    vis = oskar.vis.read(filename);
    visM = oskar.vis.to_matrix(vis);
end
