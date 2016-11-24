/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#include "ms/oskar_measurement_set.h"
#include "ms/private_ms.h"

#include <tables/Tables.h>
#include <casa/Arrays/Vector.h>

#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace casa;

static std::vector<std::string> split_string(const std::string& s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> v;
    while (std::getline(ss, item, delim))
    {
        v.push_back(item);
    }
    return v;
}

static double current_utc_to_mjd()
{
    int a, y, m, jdn;
    double day_fraction;
    time_t unix_time;
    struct tm* time_s;

    // Get system UTC.
    unix_time = std::time(NULL);
    time_s = std::gmtime(&unix_time);

    // Compute Julian Day Number (Note: all integer division).
    // Note that tm_mon is in range 0-11, so must add 1.
    a = (14 - (time_s->tm_mon + 1)) / 12;
    y = (time_s->tm_year + 1900) + 4800 - a;
    m = (time_s->tm_mon + 1) + 12 * a - 3;
    jdn = time_s->tm_mday + (153 * m + 2) / 5 + (365 * y) + (y / 4) - (y / 100)
            + (y / 400) - 32045;

    // Compute day fraction.
    day_fraction = time_s->tm_hour / 24.0 + time_s->tm_min / 1440.0 +
            time_s->tm_sec / 86400.0;
    return jdn + day_fraction - 2400000.5 - 0.5;
}

void oskar_ms_add_history(oskar_MeasurementSet* p, const char* origin,
        const char* str, size_t size)
{
    if (!p->ms || !p->msc) return;
    if (!str || size == 0) return;

    // Construct a string from the char array and split on each newline.
    std::vector<std::string> v = split_string(std::string(str, size), '\n');

    // Add to the HISTORY table.
    int num_lines = v.size();
    double current_utc = 86400.0 * current_utc_to_mjd();
    for (int i = 0; i < num_lines; ++i)
    {
        int row = p->ms->history().nrow();
        p->ms->history().addRow(1);
        MSHistoryColumns& c = p->msc->history();
        c.message().put(row, String(v[i]));
        c.application().put(row, p->app_name);
        c.priority().put(row, "INFO");
        c.origin().put(row, origin);
        c.time().put(row, current_utc);
        c.observationId().put(row, -1);
        c.appParams().put(row, Vector<String>());
        c.cliCommand().put(row, Vector<String>()); // Required!
    }
}
