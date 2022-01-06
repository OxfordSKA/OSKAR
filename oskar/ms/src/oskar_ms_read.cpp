/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "ms/oskar_measurement_set.h"
#include "ms/private_ms.h"

#include <tables/Tables.h>
#include <casa/Arrays/Vector.h>
#include <casa/Arrays/Matrix.h>

using namespace casacore;

template<typename T>
static void copy_array(const oskar_MeasurementSet* p, const char* column,
        unsigned int start_row, unsigned int num_rows,
        size_t data_size_bytes, void* data, size_t* required_size,
        int* status)
{
    try
    {
        Slice slice(start_row, num_rows, 1);
        ArrayColumn<T> ac(*(p->ms), column);
        Array<T> a = ac.getColumnRange(slice);
        *required_size = a.size() * sizeof(T);
        if (data_size_bytes >= *required_size)
        {
            memcpy(data, a.data(), *required_size);
        }
        else
        {
            *status = OSKAR_ERR_MS_OUT_OF_RANGE;
        }
    }
    catch (...)
    {
        *status = OSKAR_ERR_MS_NO_DATA;
    }
}

template<typename T>
static void copy_scalar(const oskar_MeasurementSet* p, const char* column,
        unsigned int start_row, unsigned int num_rows,
        size_t data_size_bytes, void* data, size_t* required_size,
        int* status)
{
    try
    {
        Slice slice(start_row, num_rows, 1);
        ScalarColumn<T> ac(*(p->ms), column);
        Array<T> a = ac.getColumnRange(slice);
        *required_size = a.size() * sizeof(T);
        if (data_size_bytes >= *required_size)
        {
            memcpy(data, a.data(), *required_size);
        }
        else
        {
            *status = OSKAR_ERR_MS_OUT_OF_RANGE;
        }
    }
    catch (...)
    {
        *status = OSKAR_ERR_MS_NO_DATA;
    }
}

void oskar_ms_read_column(const oskar_MeasurementSet* p, const char* column,
        unsigned int start_row, unsigned int num_rows,
        size_t data_size_bytes, void* data, size_t* required_size_bytes,
        int* status)
{
    if (*status || !p->ms) return;

    // Check that the column exists.
    if (!p->ms->tableDesc().isColumn(column))
    {
        *status = OSKAR_ERR_MS_COLUMN_NOT_FOUND;
        return;
    }

    // Check that some data are selected.
    if (num_rows == 0) return;

    // Check that the row is within the table bounds.
    unsigned int total_rows = p->ms->nrow();
    if (start_row >= total_rows)
    {
        *status = OSKAR_ERR_MS_OUT_OF_RANGE;
        return;
    }
    if (start_row + num_rows > total_rows)
    {
        num_rows = total_rows - start_row;
    }

    // Get column description and data type.
    const ColumnDesc& cdesc = p->ms->tableDesc().columnDesc(column);
    DataType dtype = cdesc.dataType();

    if (cdesc.isScalar())
    {
        switch (dtype)
        {
        case TpBool:
            copy_scalar<Bool>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpChar:
            copy_scalar<Char>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpUChar:
            copy_scalar<uChar>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpShort:
            copy_scalar<Short>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpUShort:
            copy_scalar<uShort>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpInt:
            copy_scalar<Int>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpUInt:
            copy_scalar<uInt>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpFloat:
            copy_scalar<Float>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpDouble:
            copy_scalar<Double>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpComplex:
            copy_scalar<Complex>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpDComplex:
            copy_scalar<DComplex>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        default:
            *status = OSKAR_ERR_MS_UNKNOWN_DATA_TYPE; break;
        }
    }
    else
    {
        switch (dtype)
        {
        case TpBool:
            copy_array<Bool>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpChar:
            copy_array<Char>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpUChar:
            copy_array<uChar>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpShort:
            copy_array<Short>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpUShort:
            copy_array<uShort>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpInt:
            copy_array<Int>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpUInt:
            copy_array<uInt>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpFloat:
            copy_array<Float>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpDouble:
            copy_array<Double>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpComplex:
            copy_array<Complex>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        case TpDComplex:
            copy_array<DComplex>(p, column, start_row, num_rows,
                    data_size_bytes, data, required_size_bytes, status); break;
        default:
            *status = OSKAR_ERR_MS_UNKNOWN_DATA_TYPE; break;
        }
    }
}

template <typename T>
void oskar_ms_read_coords(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int num_baselines,
        T* uu, T* vv, T* ww, int* status)
{
    if (!p->ms || num_baselines == 0) return;

    // Check that the row is within the table bounds.
    unsigned int total_rows = p->ms->nrow();
    if (start_row >= total_rows)
    {
        *status = OSKAR_ERR_MS_OUT_OF_RANGE;
        return;
    }
    if (start_row + num_baselines > total_rows)
    {
        num_baselines = total_rows - start_row;
    }

    // Read the coordinate data and copy it into the supplied arrays.
    Slice slice(start_row, num_baselines, 1);
#ifdef OSKAR_MS_NEW
    Array<Double> column_range = p->msmc.uvw.getColumnRange(slice);
#else
    Array<Double> column_range = p->msmc->uvw().getColumnRange(slice);
#endif
    Matrix<Double> matrix;
    matrix.reference(column_range);
    for (unsigned int i = 0; i < num_baselines; ++i)
    {
        uu[i] = matrix(0, i);
        vv[i] = matrix(1, i);
        ww[i] = matrix(2, i);
    }
}

void oskar_ms_read_coords_d(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int num_baselines,
        double* uu, double* vv, double* ww, int* status)
{
    oskar_ms_read_coords(p, start_row, num_baselines, uu, vv, ww, status);
}

template <typename T>
void oskar_ms_read_vis(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int start_channel,
        unsigned int num_channels, unsigned int num_baselines,
        const char* column, T* vis, int* status)
{
    if (!p->ms || num_baselines == 0 || num_channels == 0) return;

    // Check that the column exists.
    if (!p->ms->tableDesc().isColumn(column))
    {
        *status = OSKAR_ERR_MS_COLUMN_NOT_FOUND;
        return;
    }
    if (strcmp(column, "DATA") && strcmp(column, "CORRECTED_DATA") &&
            strcmp(column, "MODEL_DATA"))
    {
        *status = OSKAR_ERR_MS_COLUMN_NOT_FOUND;
        return;
    }

    // Check that the row is within the table bounds.
    unsigned int total_rows = p->ms->nrow();
    if (start_row >= total_rows)
    {
        *status = OSKAR_ERR_MS_OUT_OF_RANGE;
        return;
    }
    if (start_row + num_baselines > total_rows)
    {
        num_baselines = total_rows - start_row;
    }

    // Create the slicers for the column.
    unsigned int num_pols = p->num_pols;
    IPosition start1(1, start_row);
    IPosition length1(1, num_baselines);
    Slicer row_range(start1, length1);
    IPosition start2(2, 0, start_channel);
    IPosition length2(2, num_pols, num_channels);
    Slicer array_section(start2, length2);

    // Read the data.
    ArrayColumn<Complex> ac(*(p->ms), column);
    Array<Complex> column_range = ac.getColumnRange(row_range, array_section);

    // Copy the visibility data into the supplied array,
    // swapping baseline and channel dimensions.
    const float* in = (const float*) column_range.data();
    for (unsigned int c = 0; c < num_channels; ++c)
    {
        for (unsigned int b = 0; b < num_baselines; ++b)
        {
            for (unsigned int p = 0; p < num_pols; ++p)
            {
                unsigned int i = (num_pols * (b * num_channels + c) + p) << 1;
                unsigned int j = (num_pols * (c * num_baselines + b) + p) << 1;
                vis[j]     = in[i];
                vis[j + 1] = in[i + 1];
            }
        }
    }
}

void oskar_ms_read_vis_f(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int start_channel,
        unsigned int num_channels, unsigned int num_baselines,
        const char* column, float* vis, int* status)
{
    oskar_ms_read_vis(p, start_row, start_channel,
            num_channels, num_baselines, column, vis, status);
}
