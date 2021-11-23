#include "splines/oskar_dierckx_fpbisp.h"
#include "splines/oskar_dierckx_fpbspl.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fpbisp_f(const float *tx, int nx, const float *ty, int ny,
    const float *c, int kx, int ky, const float *x, int mx, const float *y,
    int my, float *z, float *wx, float *wy, int *lx, int *ly)
{
    /* Local variables */
    int i = 0, j = 0, l = 0, m = 0, i1 = 0, j1 = 0, l1 = 0, l2 = 0;
    int kx1 = 0, ky1 = 0, nkx1 = 0, nky1 = 0;
    float h[6];
    float tb = 0.0f, te = 0.0f, sp = 0.0f, arg = 0.0f;

    /* Parameter adjustments */
    --tx;
    --ty;
    --c;
    --lx;
    wx -= (1 + mx);
    --x;
    --ly;
    wy -= (1 + my);
    --z;
    --y;

    /* Function Body */
    kx1 = kx + 1;
    nkx1 = nx - kx1;
    tb = tx[kx1];
    te = tx[nkx1 + 1];
    l = kx1;
    for (i = 1; i <= mx; ++i)
    {
        arg = x[i];
        if (arg < tb) arg = tb;
        if (arg > te) arg = te;
        while (!(arg < tx[l + 1] || l == nkx1)) l++;
        oskar_dierckx_fpbspl_f(&tx[1], kx, arg, l, h);
        lx[i] = l - kx1;
        for (j = 1; j <= kx1; ++j)
        {
            wx[i + j * mx] = h[j - 1];
        }
    }
    ky1 = ky + 1;
    nky1 = ny - ky1;
    tb = ty[ky1];
    te = ty[nky1 + 1];
    l = ky1;
    for (i = 1; i <= my; ++i)
    {
        arg = y[i];
        if (arg < tb) arg = tb;
        if (arg > te) arg = te;
        while (!(arg < ty[l + 1] || l == nky1)) l++;
        oskar_dierckx_fpbspl_f(&ty[1], ky, arg, l, h);
        ly[i] = l - ky1;
        for (j = 1; j <= ky1; ++j)
        {
            wy[i + j * my] = h[j - 1];
        }
    }
    m = 0;
    for (i = 1; i <= mx; ++i)
    {
        l = lx[i] * nky1;
        for (i1 = 1; i1 <= kx1; ++i1)
        {
            h[i1 - 1] = wx[i + i1 * mx];
        }
        for (j = 1; j <= my; ++j)
        {
            l1 = l + ly[j];
            sp = 0.0;
            for (i1 = 1; i1 <= kx1; ++i1)
            {
                l2 = l1;
                for (j1 = 1; j1 <= ky1; ++j1)
                {
                    ++l2;
                    sp += c[l2] * h[i1 - 1] * wy[j + j1 * my];
                }
                l1 += nky1;
            }
            ++m;
            z[m] = sp;
        }
    }
}

void oskar_dierckx_fpbisp_d(const double *tx, int nx, const double *ty, int ny,
    const double *c, int kx, int ky, const double *x, int mx, const double *y,
    int my, double *z, double *wx, double *wy, int *lx, int *ly)
{
    /* Local variables */
    int i = 0, j = 0, l = 0, m = 0, i1 = 0, j1 = 0, l1 = 0, l2 = 0;
    int kx1 = 0, ky1 = 0, nkx1 = 0, nky1 = 0;
    double h[6];
    double tb = 0.0, te = 0.0, sp = 0.0, arg = 0.0;

    /* Parameter adjustments */
    --tx;
    --ty;
    --c;
    --lx;
    wx -= (1 + mx);
    --x;
    --ly;
    wy -= (1 + my);
    --z;
    --y;

    /* Function Body */
    kx1 = kx + 1;
    nkx1 = nx - kx1;
    tb = tx[kx1];
    te = tx[nkx1 + 1];
    l = kx1;
    for (i = 1; i <= mx; ++i)
    {
        arg = x[i];
        if (arg < tb) arg = tb;
        if (arg > te) arg = te;
        while (!(arg < tx[l + 1] || l == nkx1)) l++;
        oskar_dierckx_fpbspl_d(&tx[1], kx, arg, l, h);
        lx[i] = l - kx1;
        for (j = 1; j <= kx1; ++j)
        {
            wx[i + j * mx] = h[j - 1];
        }
    }
    ky1 = ky + 1;
    nky1 = ny - ky1;
    tb = ty[ky1];
    te = ty[nky1 + 1];
    l = ky1;
    for (i = 1; i <= my; ++i)
    {
        arg = y[i];
        if (arg < tb) arg = tb;
        if (arg > te) arg = te;
        while (!(arg < ty[l + 1] || l == nky1)) l++;
        oskar_dierckx_fpbspl_d(&ty[1], ky, arg, l, h);
        ly[i] = l - ky1;
        for (j = 1; j <= ky1; ++j)
        {
            wy[i + j * my] = h[j - 1];
        }
    }
    m = 0;
    for (i = 1; i <= mx; ++i)
    {
        l = lx[i] * nky1;
        for (i1 = 1; i1 <= kx1; ++i1)
        {
            h[i1 - 1] = wx[i + i1 * mx];
        }
        for (j = 1; j <= my; ++j)
        {
            l1 = l + ly[j];
            sp = 0.0;
            for (i1 = 1; i1 <= kx1; ++i1)
            {
                l2 = l1;
                for (j1 = 1; j1 <= ky1; ++j1)
                {
                    ++l2;
                    sp += c[l2] * h[i1 - 1] * wy[j + j1 * my];
                }
                l1 += nky1;
            }
            ++m;
            z[m] = sp;
        }
    }
}

#ifdef __cplusplus
}
#endif
