/*
 * Copyright (c) 2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_TelescopeLoaderHarpData.h"
#include "telescope/private_telescope.h"
#include "telescope/station/private_station.h"
#include "utility/oskar_dir.h"

using std::map;
using std::string;

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
}

TelescopeLoaderHarpData::~TelescopeLoaderHarpData()
{
}

void TelescopeLoaderHarpData::load(oskar_Telescope* telescope,
        const string& cwd, int num_subdirs,
        map<string, string>& /*filemap*/, int* status)
{
    // Get items in directory starting with root name.
    int num_items = 0;
    char** items = 0;
    oskar_dir_items(cwd.c_str(), wildcard.c_str(), 1, 0, &num_items, &items);

    if (num_items > 0)
    {
        if (num_subdirs == 0)
        {
            // Allocate space for data per frequency.
            telescope->harp_num_freq = num_items;
            telescope->harp_data = (oskar_Harp**) calloc(
                    num_items, sizeof(oskar_Harp*));
            oskar_mem_realloc(telescope->harp_freq_cpu, num_items, status);
            double* freqs = oskar_mem_double(telescope->harp_freq_cpu, status);

            // Iterate matched files.
            for (int i = 0; i < num_items; ++i)
            {
                telescope->harp_data[i] = oskar_harp_create(telescope->precision);
                oskar_harp_open_hdf5(telescope->harp_data[i],
                        get_path(cwd, items[i]).c_str(), status);
                freqs[i] = get_freq(string(items[i]));
            }
        }
        else
        {
            oskar_log_warning(0, "Ignoring HARP data in telescope model, "
                    "as the number of station directories is nonzero.");
        }
    }
    for (int i = 0; i < num_items; ++i)
    {
        free(items[i]);
    }
    free(items);
}

void TelescopeLoaderHarpData::load(oskar_Station* station,
        const string& cwd, int /*num_subdirs*/, int /*depth*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Get items in directory starting with root name.
    int num_items = 0;
    char** items = 0;
    oskar_dir_items(cwd.c_str(), wildcard.c_str(), 1, 0, &num_items, &items);

    // Allocate space for data per frequency.
    station->harp_num_freq = num_items;
    station->harp_data = (oskar_Harp**) calloc(num_items, sizeof(oskar_Harp*));
    oskar_mem_realloc(station->harp_freq_cpu, num_items, status);
    double* freqs = oskar_mem_double(station->harp_freq_cpu, status);

    // Iterate matched files.
    for (int i = 0; i < num_items; ++i)
    {
        station->harp_data[i] = oskar_harp_create(station->precision);
        oskar_harp_open_hdf5(station->harp_data[i],
                get_path(cwd, items[i]).c_str(), status);
        freqs[i] = get_freq(string(items[i]));
        free(items[i]);
    }
    free(items);
}

string TelescopeLoaderHarpData::name() const
{
    return string("HARP data loader");
}
