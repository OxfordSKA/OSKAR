function data = read(filename)

% Query the file to obtain a table describing the records
records = oskar.binary_file.query(filename);

% Get a list of groups from the group column of the table. 
groups = cell2mat(records(2:end,3)); % list of group ids

% Find indices where the group = 8 (== GROUP_TAG_TIME_FREQ_DATA)
indices = find(groups == 8); % indices of group 8 records

% Update the record table to contain just group 8
records = records(indices+1, :);

% Find the number of unique indices.
num_idx = length(unique(cell2mat(records(:, 5))));
fprintf('Number of time-frequency indices = %i\n', num_idx);

% Read the data into an array of structures.

data = struct([]);

grp = 8;
for k = 1:num_idx
    idx = k-1;
    % Read the group.
    g = oskar.binary_file.read_group('temp_pp.dat', grp, idx);
    
    % loop over tags in the group.
    for t=1:length(g)
        r = g(t);
        % Swtich to find the 6 header tag values (these always exist)
        switch (r.tag)
            case 0 % TIME_IDX
                data(k).time_idx = r.data;
            case 1 % FREQ_IDX
                data(k).freq_idx = r.data;
            case 2 % TIME_MJD_UTC
                data(k).time_mjd_utc = r.data;
            case 3 % FREQ_HZ 
                data(k).freq_hz = r.data;
            case 4 % NUM_FIELDS
                num_fields = r.data;
            case 5 % NUM_FIELD_TAGS
                num_field_tags = r.data;
        end    
    end 
    
    
    % There are 6 header tags (0-5).
    % data field tags start at 10 and have num_field_tags tags each.
    % data tags therefore have the ID:
    %    (10 + num_field_tags * field) + 0
    % 
    data(k).field = struct([]);
    
    % Loop over tags again to find data tags.
    for t=1:length(g)
        r = g(t);
        for f = 1:num_fields
            field_offset = 10 + (num_field_tags * (f-1));
            switch (r.tag) 
                case field_offset+0 % VALUES
                    data(k).field(f).values = r.data;
                case field_offset+1 % DIMS
                    data(k).field(f).dims = r.data;
                case field_offset+2 % LABEL
                    data(k).field(f).label = r.data;
                case field_offset+3 % UNITS
                    data(k).field(f).units = r.data;
            end
        end
    end 
    
end

end

% for t = 1:length(data)
%     x = data(t).field(1).values.*(180./pi);
%     y = data(t).field(2).values.*(180./pi);
%     scatter(x, y ,'.');
%     %errorbar(x, y, ones(length(x),1).*0.1);
%     axis([-10 10 40 60]);
%     axis square;
%     %axis equal;
%     grid on;
%     title([num2str(t) ' - ' num2str(data(t).time_mjd_utc)]);
%     xlabel([data(t).field(1).label ' (deg)'], 'interpreter','none');
%     ylabel([data(t).field(2).label ' (deg)'], 'interpreter','none');
%     pause(0.8);
% end

