#include "splines/oskar_dierckx_fpgivs.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fpgivs(double piv, double *ww, double *cos_, double *sin_)
{
    double dd = 0.0, store = 0.0, t = 0.0;
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
