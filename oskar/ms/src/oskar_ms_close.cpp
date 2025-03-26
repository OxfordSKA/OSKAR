/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "ms/oskar_measurement_set.h"
#include "ms/private_ms.h"

#include <cstdlib>

void oskar_ms_close(oskar_MeasurementSet* p)
{
    if (!p) return;
    if (p->data_written)
    {
        oskar_ms_set_time_range(p);
    }
    if (p->ms)
    {
        delete p->ms;
    }
    free(p->a1);
    free(p->a2);
    free(p->app_name);
    free(p);
}
