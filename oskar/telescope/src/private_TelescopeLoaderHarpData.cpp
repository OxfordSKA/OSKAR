/*
 * Copyright (c) 2022-2024, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdlib>

#include "oskar/telescope/private_TelescopeLoaderHarpData.h"
#include "oskar/telescope/private_telescope.h"
#include "oskar/telescope/station/private_station.h"
#include "oskar/utility/oskar_dir.h"

using std::map;
using std::string;
using std::vector;

static const char root_name[] = "HARP";

static string remove_extension(const string& filename)
{
    size_t dot_position = filename.find_last_of(".");
    if (dot_position == string::npos) return filename;
    return filename.substr(0, dot_position);
}

static double get_freq(const string& str)
{
    string without_extension = remove_extension(str);
    string stripped = without_extension.substr(
            0, without_extension.find_last_of(".0123456789") + 1);
    size_t digit_position = stripped.find_last_not_of(".0123456789") + 1;
    return 1e6 * strtod(stripped.c_str() + digit_position, 0);
}

TelescopeLoaderHarpData::TelescopeLoaderHarpData()
{
    wildcard = string("*") + string(root_name) + string("*");
    mutex = oskar_mutex_create();
}

TelescopeLoaderHarpData::~TelescopeLoaderHarpData()
{
    for (map<string, oskar_Harp*>::iterator i = model_map.begin();
            i != model_map.end(); ++i)
    {
        oskar_harp_ref_dec(i->second);
    }
    oskar_mutex_free(mutex);
}

void TelescopeLoaderHarpData::load(oskar_Telescope* telescope,
        const string& cwd, int num_subdirs, map<string, string>& filemap,
        int* status)
{
    update_map(filemap, cwd);

    if (num_subdirs == 0)
    {
        vector<string> items = get_path_list(filemap, status);
        int num_items = (int)items.size();
        if (num_items > 0)
        {
            // Special case to allow individual elements to be correlated.
            // Allocate space for data per frequency.
            telescope->harp_num_freq = num_items;
            telescope->harp_data = (oskar_Harp**) calloc(
                    num_items, sizeof(oskar_Harp*));
            oskar_mem_realloc(telescope->harp_freq_cpu, num_items, status);
            double* freqs = oskar_mem_double(telescope->harp_freq_cpu, status);

            // Iterate matched files.
            oskar_mutex_lock(mutex);
            for (int i = 0; i < num_items; ++i)
            {
                telescope->harp_data[i] = oskar_harp_ref_inc(
                        get_harp_model(items[i], telescope->precision)
                );
                freqs[i] = get_freq(items[i]);
            }
            oskar_mutex_unlock(mutex);
        }
    }
}

void TelescopeLoaderHarpData::load(oskar_Station* station,
        const string& cwd, int num_subdirs, int /*depth*/,
        map<string, string>& filemap, int* status)
{
    update_map(filemap, cwd);

    if (num_subdirs == 0)
    {
        vector<string> items = get_path_list(filemap, status);
        int num_items = (int)items.size();
        if (num_items > 0)
        {
            // Allocate space for data per frequency.
            station->harp_num_freq = num_items;
            station->harp_data = (oskar_Harp**) calloc(
                    num_items, sizeof(oskar_Harp*));
            oskar_mem_realloc(station->harp_freq_cpu, num_items, status);
            double* freqs = oskar_mem_double(station->harp_freq_cpu, status);

            // Iterate matched files.
            oskar_mutex_lock(mutex);
            for (int i = 0; i < num_items; ++i)
            {
                station->harp_data[i] = oskar_harp_ref_inc(
                        get_harp_model(items[i], station->precision)
                );
                freqs[i] = get_freq(items[i]);
            }
            oskar_mutex_unlock(mutex);
        }
    }
}

string TelescopeLoaderHarpData::name() const
{
    return string("HARP data loader");
}

oskar_Harp* TelescopeLoaderHarpData::get_harp_model(const string& path,
        int precision)
{
    // Check to see if an in-memory model already exists.
    map<string, oskar_Harp*>::iterator it = model_map.find(path);
    if (it != model_map.end())
    {
        // If it's already known, return a pointer to it.
        return it->second;
    }
    else
    {
        // Otherwise, create a new model and store it in the map,
        // with its full path as the key.
        oskar_Harp* harp_data = oskar_harp_create(precision);
        oskar_harp_set_file(harp_data, path.c_str());
        model_map[path] = harp_data;
        return harp_data;
    }
}

vector<string> TelescopeLoaderHarpData::get_path_list(
        const map<string, string>& filemap, int* status)
{
    vector<string> paths;
    if (*status) return paths;
    for (map<string, string>::const_iterator i = filemap.begin();
            i != filemap.end(); ++i)
    {
        string key = i->first;
        if (key.find("HARP", 0) != string::npos)
        {
            paths.push_back(i->second);
        }
    }
    return paths;
}

void TelescopeLoaderHarpData::update_map(
        map<string, string>& filemap, const string& cwd)
{
    // Get a list of files in the current directory that match the wildcard.
    // Store their full paths in the map, with the local filename as the key.
    // Any coefficients present will override ones from a higher level.
    int num_items = 0;
    char** items = 0;
    oskar_dir_items(cwd.c_str(), wildcard.c_str(), 1, 0, &num_items, &items);
    oskar_mutex_lock(mutex);
    for (int i = 0; i < num_items; ++i)
    {
        filemap[items[i]] = get_path(cwd, string(items[i]));
        free(items[i]);
    }
    oskar_mutex_unlock(mutex);
    free(items);
}
