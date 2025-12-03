/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "sky/oskar_sky.h"
#include "sky/private_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_free(oskar_Sky* model, int* status)
{
    int i = 0;
    if (!model) return;
    const int num_columns = model->attr_int[OSKAR_SKY_NUM_COLUMNS];
    for (i = 0; i < num_columns; ++i)
    {
        oskar_mem_free(model->columns[i], status);
    }
    oskar_mem_free(model->table, status);
    free(model->columns);
    free(model->column_attr);
    free(model->column_type);
    free(model->attr_double);
    free(model->attr_int);
    free(model);
}

#ifdef __cplusplus
}
#endif
