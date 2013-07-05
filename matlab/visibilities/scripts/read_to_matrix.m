function visM = read_to_matrix(filename)
    vis = oskar.visibilities.read(filename);
    visM = oskar.visibilities.to_matrix(vis);
end
