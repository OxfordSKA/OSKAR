/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_date_time_to_mjd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Note: consider returning separate day and day fraction to gain a bit more
 * floating point accuracy.
 */
double oskar_convert_date_time_to_mjd(int year, int month, int day,
        double day_fraction)
{
    int a = 0, y = 0, m = 0, jdn = 0;

    /* Compute Julian Day Number (Note: all integer division). */
    a = (14 - month) / 12;
    y = year + 4800 - a;
    m = month + 12 * a - 3;
    jdn = day + (153 * m + 2) / 5 + (365 * y) + (y / 4) - (y / 100)
            + (y / 400) - 32045;

    /* Compute day fraction. */
    day_fraction -= 0.5;
    return (jdn - 2400000.5) + day_fraction;
}

#ifdef __cplusplus
}
#endif
