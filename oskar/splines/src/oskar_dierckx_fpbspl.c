#include "splines/oskar_dierckx_fpbspl.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fpbspl_f(const float *t, int k, float x, int l, float *h)
{
    /* Local variables */
    float f = 0.0f, hh[5];
    int i = 0, j = 0, li = 0, lj = 0;

    /* Parameter adjustments */
    --t;
    --h;

    /* Function Body */
    h[1] = 1.0;
    for (j = 1; j <= k; ++j)
    {
        for (i = 1; i <= j; ++i)
        {
            hh[i - 1] = h[i];
        }
        h[1] = 0.0;
        for (i = 1; i <= j; ++i)
        {
            li = l + i;
            lj = li - j;
            f = hh[i - 1] / (t[li] - t[lj]);
            h[i] += f * (t[li] - x);
            h[i + 1] = f * (x - t[lj]);
        }
    }
}

void oskar_dierckx_fpbspl_d(const double *t, int k, double x, int l, double *h)
{
    /* Local variables */
    double f = 0.0, hh[5];
    int i = 0, j = 0, li = 0, lj = 0;

    /* Parameter adjustments */
    --t;
    --h;

    /* Function Body */
    h[1] = 1.0;
    for (j = 1; j <= k; ++j)
    {
        for (i = 1; i <= j; ++i)
        {
            hh[i - 1] = h[i];
        }
        h[1] = 0.0;
        for (i = 1; i <= j; ++i)
        {
            li = l + i;
            lj = li - j;
            f = hh[i - 1] / (t[li] - t[lj]);
            h[i] += f * (t[li] - x);
            h[i + 1] = f * (x - t[lj]);
        }
    }
}

#ifdef __cplusplus
}
#endif
