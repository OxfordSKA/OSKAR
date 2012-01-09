#include "extern/dierckx/fpgivs.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void fpgivs(float piv, float *ww, float *cos_, float *sin_)
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

#ifdef __cplusplus
}
#endif
