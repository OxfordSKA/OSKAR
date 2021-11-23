/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_mjd_utc_to_mjd_tt.h"

#ifdef __cplusplus
extern "C" {
#endif

double oskar_convert_mjd_utc_to_mjd_tt(double mjd_utc,
        double delta_tai_utc_sec)
{
    double delta = 0.0;

    /* Get total delta in days. */
    delta = delta_tai_utc_sec + 32.184;
    delta /= 86400.0;

    /* MJD(TT) = MJD(UTC) + delta_tai_utc + 32.184 sec. */
    return mjd_utc + delta;
}

#ifdef __cplusplus
}
#endif
