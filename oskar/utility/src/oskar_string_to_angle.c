/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "math/oskar_cmath.h"
#include "utility/oskar_string_to_angle.h"

#define DEG2RAD (M_PI / 180.0)

#ifdef __cplusplus
extern "C" {
#endif


double oskar_string_degrees_to_radians(
        const char* str,
        char default_unit,
        int* status
)
{
    int dot_count = 0;
    int d = 0, m = 0;
    double sec = 0.0;
    int use_sexagesimal = 0;
    const char* p = str;
    if (!str || !*str) return 0.0;
    while (*p && isspace(*p)) p++;
    for (const char* t = p; *t; ++t)
    {
        if (*t == '.') dot_count++;
    }
    if (dot_count >= 2)
    {
        if (sscanf(p, "%d.%d.%lf", &d, &m, &sec) != 3)
        {
            *status = OSKAR_ERR_BAD_UNITS;
        }
        use_sexagesimal = 1;
    }
    else if (strchr(p, ':'))
    {
        /* Not allowed in LOFAR-format sky models, but accepted here. */
        if (sscanf(p, "%d:%d:%lf", &d, &m, &sec) != 3)
        {
            *status = OSKAR_ERR_BAD_UNITS;
        }
        use_sexagesimal = 1;
    }
    else if (strchr(p, 'd') && strchr(p, 'm'))
    {
        if (sscanf(p, "%dd%dm%lf", &d, &m, &sec) != 3)
        {
            *status = OSKAR_ERR_BAD_UNITS;
        }
        use_sexagesimal = 1;
    }
    if (use_sexagesimal)
    {
        const double sign = (p[0] == '-') ? -1.0 : 1.0;
        const double deg = sign * (fabs((double) d) + m / 60.0 + sec / 3600.0);
        if (m >= 60 || sec >= 60.0) *status = OSKAR_ERR_BAD_UNITS;
        return deg * DEG2RAD;
    }
    const double val = strtod(p, 0);
    if (strstr(p, "deg")) return val * DEG2RAD;
    if (strstr(p, "rad")) return val;
    return tolower(default_unit) == 'r' ? val : val * DEG2RAD;
}


double oskar_string_hours_to_radians(
        const char* str,
        char default_unit,
        int* status
)
{
    int h = 0, m = 0;
    double sec = 0.0;
    int use_sexagesimal = 0;
    const char* p = str;
    if (!str || !*str) return 0.0;
    while (*p && isspace(*p)) p++;
    if (strchr(p, ':'))
    {
        if (sscanf(p, "%d:%d:%lf", &h, &m, &sec) != 3)
        {
            *status = OSKAR_ERR_BAD_UNITS;
        }
        use_sexagesimal = 1;
    }
    else if (strchr(p, 'h') && strchr(p, 'm'))
    {
        if (sscanf(p, "%dh%dm%lf", &h, &m, &sec) != 3)
        {
            *status = OSKAR_ERR_BAD_UNITS;
        }
        use_sexagesimal = 1;
    }
    if (use_sexagesimal)
    {
        const double sign = 15.0 * ((p[0] == '-') ? -1.0 : 1.0);
        const double deg = sign * (fabs((double) h) + m / 60.0 + sec / 3600.0);
        if (m >= 60 || sec >= 60.0) *status = OSKAR_ERR_BAD_UNITS;
        return deg * DEG2RAD;
    }
    const double val = strtod(p, 0);
    if (strstr(p, "deg")) return val * DEG2RAD;
    if (strstr(p, "rad")) return val;
    return tolower(default_unit) == 'r' ? val : val * DEG2RAD;
}

#ifdef __cplusplus
}
#endif
