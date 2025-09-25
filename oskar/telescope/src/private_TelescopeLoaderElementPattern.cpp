/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <sstream>
#include <cstdlib>
#include <cstring>

#include "telescope/private_TelescopeLoaderElementPattern.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_file_exists.h"

using std::vector;
using std::map;
using std::string;


TelescopeLoaderElementPattern::TelescopeLoaderElementPattern()
{
    telescope_ = 0;
    max_order_ = 0;
}


TelescopeLoaderElementPattern::~TelescopeLoaderElementPattern()
{
}


void TelescopeLoaderElementPattern::load(
        oskar_Telescope* telescope,
        const string& cwd,
        int num_subdirs,
        map<string, string>& filemap,
        int* status
)
{
    telescope_ = telescope;
    max_order_ = oskar_telescope_spherical_wave_max_order(telescope);

    update_map(filemap, cwd);

    // Load element pattern data for stations only at the deepest level!
    if (num_subdirs == 0)
    {
        const int num_stations = oskar_telescope_num_station_models(telescope);
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* s = oskar_telescope_station(telescope, i);
            oskar_station_resize_element_types(s, 1, status);
            load_element_patterns(s, filemap, status);
        }
    }
}


void TelescopeLoaderElementPattern::load(
        oskar_Station* station,
        const string& cwd,
        int num_subdirs,
        int /*depth*/,
        map<string, string>& filemap,
        int* status
)
{
    update_map(filemap, cwd);

    // Load element pattern data for stations only at the deepest level!
    if (num_subdirs == 0)
    {
        oskar_station_resize_element_types(station, 1, status);
        load_element_patterns(station, filemap, status);
    }
}


string TelescopeLoaderElementPattern::name() const
{
    return string("element pattern loader");
}


int TelescopeLoaderElementPattern::ends_with(
        const char* str,
        const char* suffix
)
{
    const size_t str_len = strlen(str);
    const size_t suffix_len = strlen(suffix);
    if (suffix_len > str_len) return 0;
    return !strncmp(str + str_len - suffix_len, suffix, suffix_len);
}


void TelescopeLoaderElementPattern::load_element_patterns(
        oskar_Station* station,
        const map<string, string>& filemap,
        int* status
)
{
    if (*status) return;

    // FIXME(FD) Return if element patterns are disabled.
    // FIXME(FD) Don't do this here, if loading functional data...
    if (!oskar_telescope_enable_numerical_patterns(telescope_)) return;

    // Get lists of all paths in the map that have keys starting with the
    // right root name.
    vector<string> keys_sw_fit, paths_sw_fit;
    vector<string> keys_sw_feko, paths_sw_feko;
    vector<string> keys_sw_feko_h5, paths_sw_feko_h5;
    vector<string> keys_sw_galileo, paths_sw_galileo;
    for (map<string, string>::const_iterator i = filemap.begin();
            i != filemap.end(); ++i)
    {
        string key = i->first;
        if (ends_with(key.c_str(), "txt"))
        {
            if (key.find("_feko", 0) != string::npos)
            {
                keys_sw_feko.push_back(key);
                paths_sw_feko.push_back(i->second);
            }
            else if (key.find("_galileo", 0) != string::npos)
            {
                keys_sw_galileo.push_back(key);
                paths_sw_galileo.push_back(i->second);
            }
            else if (key.find("_wave", 0) != string::npos)
            {
                keys_sw_fit.push_back(key);
                paths_sw_fit.push_back(i->second);
            }
        }
        else if (ends_with(key.c_str(), "h5"))
        {
            if (key.find("_FEKO", 0) != string::npos)
            {
                keys_sw_feko_h5.push_back(key);
                paths_sw_feko_h5.push_back(i->second);
            }
        }
    }

    // Load fitted X, Y or scalar data.
    load_spherical_wave_data(station, keys_sw_fit, paths_sw_fit, status);
    load_spherical_wave_feko_data(station, keys_sw_feko, paths_sw_feko, status);
    load_spherical_wave_galileo_data(
            station, keys_sw_galileo, paths_sw_galileo, status
    );
    load_spherical_wave_feko_h5_data(
            station, keys_sw_feko_h5, paths_sw_feko_h5, status
    );
}


void TelescopeLoaderElementPattern::load_spherical_wave_data(
        oskar_Station* station,
        const vector<string>& keys,
        const vector<string>& paths,
        int* status
)
{
    if (*status || !station || keys.empty()) return;
    size_t buflen = 0;
    char* buffer = 0;
    int num_tmp = 21;
    double* tmp = (double*) calloc((size_t) num_tmp, sizeof(double));
    for (size_t i = 0; i < keys.size(); ++i)
    {
        int ind = 0;
        double freq = 0.0;

        // Get the element index and frequency from the key.
        parse_filename(keys[i].c_str(), &buffer, &buflen, &ind, &freq, status);

        // Load the file.
        if (*status) break;
        if (oskar_station_num_element_types(station) < ind + 1)
        {
            oskar_station_resize_element_types(station, ind + 1, status);
        }
        oskar_element_load_spherical_wave_coeff(
                oskar_station_element(station, ind), paths[i].c_str(),
                max_order_, freq, &num_tmp, &tmp, status
        );
    }
    free(buffer);
    free(tmp);
}


void TelescopeLoaderElementPattern::load_spherical_wave_feko_data(
        oskar_Station* station,
        const vector<string>& keys,
        const vector<string>& paths,
        int* status
)
{
    if (*status || !station || keys.empty()) return;
    size_t buflen = 0;
    char* buffer = 0;
    int num_tmp = 21;
    double* tmp = (double*) calloc((size_t) num_tmp, sizeof(double));
    for (size_t i = 0; i < keys.size(); ++i)
    {
        int ind = 0;
        double freq = 0.0;

        // Get the element index and frequency from the key.
        parse_filename(keys[i].c_str(), &buffer, &buflen, &ind, &freq, status);

        // Load the file.
        if (*status) break;
        if (oskar_station_num_element_types(station) < ind + 1)
        {
            oskar_station_resize_element_types(station, ind + 1, status);
        }
        oskar_element_load_spherical_wave_coeff_feko(
                oskar_station_element(station, ind), paths[i].c_str(),
                max_order_, freq, &num_tmp, &tmp, status
        );
    }
    free(buffer);
    free(tmp);
}


void TelescopeLoaderElementPattern::load_spherical_wave_feko_h5_data(
        oskar_Station* station,
        const vector<string>& keys,
        const vector<string>& paths,
        int* status
)
{
    if (*status || !station || keys.empty()) return;
    for (size_t i = 0; i < keys.size(); ++i)
    {
        oskar_station_load_spherical_wave_coeff_feko_h5(
                station, paths[i].c_str(), max_order_, status
        );
    }
}


void TelescopeLoaderElementPattern::load_spherical_wave_galileo_data(
        oskar_Station* station,
        const vector<string>& keys,
        const vector<string>& paths,
        int* status
)
{
    if (*status || !station || keys.empty()) return;
    size_t buflen = 0;
    char* buffer = 0;
    int num_tmp = 21;
    double* tmp = (double*) calloc((size_t) num_tmp, sizeof(double));
    for (size_t i = 0; i < keys.size(); ++i)
    {
        int ind = 0;
        double freq = 0.0;

        // Get the element index and frequency from the key.
        parse_filename(keys[i].c_str(), &buffer, &buflen, &ind, &freq, status);

        // Load the file.
        if (*status) break;
        if (oskar_station_num_element_types(station) < ind + 1)
        {
            oskar_station_resize_element_types(station, ind + 1, status);
        }
        oskar_element_load_spherical_wave_coeff_galileo(
                oskar_station_element(station, ind), paths[i].c_str(),
                max_order_, freq, &num_tmp, &tmp, status
        );
    }
    free(buffer);
    free(tmp);
}


void TelescopeLoaderElementPattern::parse_filename(
        const char* s,
        char** buffer,
        size_t* buflen,
        int* index,
        double* freq,
        int* status
)
{
    size_t i = 0, j = 1, length = 0;
    char *p = 0, *end_ptr = 0;
    const int base = 10;
    *index = 0;
    if (freq) *freq = 0.0;

    // Ensure buffer is big enough to make a copy of the input string.
    length = 1 + strlen(s);
    if (*buflen < length)
    {
        char* new_buffer = (char*) realloc((void*)(*buffer), length);
        if (!new_buffer)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }
        *buffer = new_buffer;
        *buflen = length;
    }
    memcpy(*buffer, s, length);

    // Replace underscores with NULL and find out how many words there are.
    for (i = 0; i < length; ++i)
    {
        if ((*buffer)[i] == '_')
        {
            (*buffer)[i] = 0;
            j++;
        }
    }

    // Extract index and frequency as the first two numbers.
    // The frequency need not exist.
    p = *buffer;
    for (i = 0; i < j; ++i)
    {
        length = 1 + strlen(p);
        *index = (int) strtol(p, &end_ptr, base);
        if (end_ptr > p)
        {
            if ((i + 1 < j) && freq)
            {
                p += length;
                *freq = 1e6 * strtod(p, 0); // Convert from MHz to Hz.
            }
            return;
        }
        p += length;
    }
}


void TelescopeLoaderElementPattern::update_map(
        map<string, string>& filemap,
        const string& cwd
)
{
    // Update the map of element files for the current directory.
    // The presence of coefficients is sufficient to override ones
    // from a higher level.

    // Get a listing of the files in the current directory that match
    // element pattern data.
    // Store the full paths to these files in the persistent map, with the
    // local filename as the key.
    {
        // Element pattern text files.
        int num_items = 0;
        char** items = 0;
        oskar_dir_items(
                cwd.c_str(), "element_pattern*", 1, 0, &num_items, &items
        );
        for (int i = 0; i < num_items; ++i)
        {
            filemap[items[i]] = get_path(cwd, string(items[i]));
            free(items[i]);
        }
        free(items);
    }
    {
        // FEKO HDF5 files.
        int num_items = 0;
        char** items = 0;
        oskar_dir_items(cwd.c_str(), "*FEKO*.h5", 1, 0, &num_items, &items);
        for (int i = 0; i < num_items; ++i)
        {
            filemap[items[i]] = get_path(cwd, string(items[i]));
            free(items[i]);
        }
        free(items);
    }
}
