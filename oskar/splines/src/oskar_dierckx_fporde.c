#include "splines/oskar_dierckx_fporde.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fporde(const double *x, const double *y, int m, int kx,
        int ky, const double *tx, int nx, const double *ty, int ny,
        int *nummer, int *index, int nreg)
{
    /* Local variables */
    int i = 0, k = 0, l = 0, im = 0, kx1 = 0, ky1 = 0;
    int num = 0, nyy = 0, nk1x = 0, nk1y = 0;
    double xi = 0.0, yi = 0.0;

    /* Parameter adjustments */
    --nummer;
    --y;
    --x;
    --tx;
    --ty;
    --index;

    /* Function Body */
    kx1 = kx + 1;
    ky1 = ky + 1;
    nk1x = nx - kx1;
    nk1y = ny - ky1;
    nyy = nk1y - ky;
    for (i = 1; i <= nreg; ++i)
    {
        index[i] = 0;
    }
    for (im = 1; im <= m; ++im)
    {
        xi = x[im];
        yi = y[im];
        l = kx1;
        k = ky1;
        while (!(xi < tx[l + 1] || l == nk1x)) l++;
        while (!(yi < ty[k + 1] || k == nk1y)) k++;
        num = (l - kx1) * nyy + k - ky;
        nummer[im] = index[num];
        index[num] = im;
    }
}

#ifdef __cplusplus
}
#endif
