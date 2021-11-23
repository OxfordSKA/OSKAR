#include "splines/oskar_dierckx_fprati.h"

#ifdef __cplusplus
extern "C" {
#endif

double oskar_dierckx_fprati(double *p1, double *f1, double p2, double f2,
        double *p3, double *f3)
{
    /* Local variables */
    double p = 0.0, h1 = 0.0, h2 = 0.0, h3 = 0.0;

    if (*p3 > 0.0)
    {
        /* value of p in case p3 ^= infinity. */
        h1 = *f1 * (f2 - *f3);
        h2 = f2 * (*f3 - *f1);
        h3 = *f3 * (*f1 - f2);
        p = -(*p1 * p2 * h3 + p2 * *p3 * h1 + *p3 * *p1 * h2) /
                (*p1 * h1 + p2 * h2 + *p3 * h3);
    }
    else
    {
        /* value of p in case p3 = infinity. */
        p = (*p1 * (*f1 - *f3) * f2 - p2 * (f2 - *f3) * *f1) /
                ((*f1 - f2) * *f3);
    }

    /* adjust the value of p1,f1,p3 and f3 such that f1 > 0 and f3 < 0. */
    if (f2 < 0.0)
    {
        *p3 = p2;
        *f3 = f2;
    }
    else
    {
        *p1 = p2;
        *f1 = f2;
    }
    return p;
}

#ifdef __cplusplus
}
#endif
