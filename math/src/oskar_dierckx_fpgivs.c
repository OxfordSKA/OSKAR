#include "math/oskar_dierckx_fpgivs.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fpgivs_f(float piv, float *ww, float *cos_, float *sin_)
{
    float dd, store, t;
    store = fabsf(piv);
    if (store < *ww)
    {
        t = piv / *ww;
        dd = *ww * sqrtf(1.0 + t * t);
    }
    else
    {
        t = *ww / piv;
        dd = store * sqrtf(1.0 + t * t);
    }
    *cos_ = *ww / dd;
    *sin_ = piv / dd;
    *ww = dd;
}

void oskar_dierckx_fpgivs_d(double piv, double *ww, double *cos_, double *sin_)
{
    double dd, store, t;
    store = fabs(piv);
    if (store < *ww)
    {
        t = piv / *ww;
        dd = *ww * sqrt(1.0 + t * t);
    }
    else
    {
        t = *ww / piv;
        dd = store * sqrt(1.0 + t * t);
    }
    *cos_ = *ww / dd;
    *sin_ = piv / dd;
    *ww = dd;
}

#ifdef __cplusplus
}
#endif
