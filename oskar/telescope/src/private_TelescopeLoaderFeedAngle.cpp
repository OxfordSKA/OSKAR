/*
 * Copyright (c) 2015-2020, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "telescope/private_TelescopeLoaderFeedAngle.h"
#include "utility/oskar_dir.h"

using std::map;
using std::string;

static const char* feed_angle_file = "feed_angle.txt";
static const char* feed_angle_file_x = "feed_angle_x.txt";
static const char* feed_angle_file_y = "feed_angle_y.txt";

void TelescopeLoaderFeedAngle::load(oskar_Station* station,
        const string& cwd, int /*num_subdirs*/, int /*depth*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of feed angle files.
    if (oskar_dir_file_exists(cwd.c_str(), feed_angle_file))
    {
        string f = get_path(cwd, feed_angle_file);
        oskar_station_load_feed_angle(station, 0, f.c_str(), status);
        oskar_station_load_feed_angle(station, 1, f.c_str(), status);
    }
    if (oskar_dir_file_exists(cwd.c_str(), feed_angle_file_x))
    {
        oskar_station_load_feed_angle(station, 0,
                get_path(cwd, feed_angle_file_x).c_str(), status);
    }
    if (oskar_dir_file_exists(cwd.c_str(), feed_angle_file_y))
    {
        oskar_station_load_feed_angle(station, 1,
                get_path(cwd, feed_angle_file_y).c_str(), status);
    }
}

string TelescopeLoaderFeedAngle::name() const
{
    return string("element feed angle file loader");
}
