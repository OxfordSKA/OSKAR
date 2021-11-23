/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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

using namespace casacore;

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
    int a = 0, y = 0, m = 0, jdn = 0;
    double day_fraction = 0.0;
    time_t unix_time = 0;
    struct tm* time_s = 0;

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
    if (!p->ms) return;
    if (!str || size == 0) return;

    // Construct a string from the char array and split on each newline.
    std::vector<std::string> v = split_string(std::string(str, size), '\n');

    // Add to the HISTORY table.
#ifdef OSKAR_MS_NEW
    Table history(p->ms->tableName() + "/HISTORY", Table::Update);
    ScalarColumn<String> message(history, "MESSAGE");
    ScalarColumn<String> application(history, "APPLICATION");
    ScalarColumn<String> priority(history, "PRIORITY");
    ScalarColumn<String> originCol(history, "ORIGIN");
    ScalarColumn<Double> timeCol(history, "TIME");
    ScalarColumn<Int> observationId(history, "OBSERVATION_ID");
    ArrayColumn<String> appParams(history, "APP_PARAMS");
    ArrayColumn<String> cliCommand(history, "CLI_COMMAND");
#endif
    int num_lines = v.size();
    double current_utc = 86400.0 * current_utc_to_mjd();
    for (int i = 0; i < num_lines; ++i)
    {
#ifdef OSKAR_MS_NEW
        int row = history.nrow();
        history.addRow(1);
        message.put(row, String(v[i]));
        application.put(row, p->app_name);
        priority.put(row, "INFO");
        originCol.put(row, origin);
        timeCol.put(row, current_utc);
        observationId.put(row, -1);
        appParams.put(row, Vector<String>());
        cliCommand.put(row, Vector<String>()); // Required!
#else
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
#endif
    }
}
