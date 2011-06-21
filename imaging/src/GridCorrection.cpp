#include "imaging/GridCorrection.h"
#include "imaging/ConvFunc.h"
#include "imaging/floating_point_compare.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

namespace oskar {

void GridCorrection::computeCorrection(ConvFunc & c, const unsigned grid_size)
{
    const unsigned c_size = c.size();
    const unsigned c_oversample = c.oversample();
    const float * convFunc = c.values();
    vector<float> c_x(c_size);
    const unsigned c_centre = c_size / 2;
    const float c_inc = 1.0f / c_oversample;

    const int g_centre = grid_size / 2;
    const float g_inc = 1.0f / static_cast<float>(grid_size);

//    printf("c_size       = %d\n", c_size);
//    printf("c_centre     = %d\n", c_centre);
//    printf("c_oversample = %d\n", c_oversample);
//    printf("c_inc        = %f\n", c_inc);

    for (unsigned i = 0; i < c_size; ++i)
    {
        c_x[i] = (float(i) - float(c_centre)) * c_inc;
//        printf("c_x = %f\n", c_x[i]);
    }

    _correction.resize(grid_size, 0.0f);
    _size = grid_size;
    float * correction = &_correction[0];

    const float f = 1.0f / float(c_oversample);

    for (unsigned j = 0; j < grid_size; ++j)
    {
        const float x = ((float)j - g_centre) / float(grid_size);
        for (unsigned i = 0; i < c_size; ++i)
        {
            const float arg = 2 * M_PI * x * c_x[i];
            correction[j] += convFunc[i] * cos(arg);
        }
    }

//    for (int i = 0; i < static_cast<int>(grid_size); ++i)
//    {
//        const float x = (static_cast<float>(i - g_centre)) * g_inc;
//        const float abs_x = fabs(x);
////        printf("x = %f %f\n", x, abs_x);
//
//        if (isEqual<float>(abs_x, 0.0f))
//            correction[i] *= 1.0f;
//        else
//        {
//            const float arg = M_PI * abs_x * f;
//            correction[i] *= sin(arg) / arg;
//        }
//    }

    float max = findMax();
    for (unsigned i = 0; i < grid_size; ++i)
    {
        correction[i] /= max;
    }
}

void GridCorrection::make2D()
{
    vector<float> temp(_size * _size);
    float * t = &temp[0];
    float * c = &_correction[0];
    for (unsigned j = 0; j < _size; ++j)
    {
        for (unsigned i = 0; i < _size; ++i)
        {
            t[j * _size + i] = c[j] * c[i];
        }
    }
    _correction.resize(_size * _size);
    memcpy((void*)&_correction[0], (const void*)t, _size * _size * sizeof(float));
}


float GridCorrection::findMax()
{
    float convmax = -numeric_limits<float>::max();
//    for (unsigned i = 0; i < _size * _size; ++i)
//        convmax = max(convmax, abs(_correction[i]));
    for (unsigned i = 0; i < _size; ++i)
        convmax = max(convmax, abs(_correction[i]));
    return convmax;
}


} // namespace oskar
