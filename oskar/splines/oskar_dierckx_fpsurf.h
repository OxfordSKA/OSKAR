#ifndef OSKAR_DIERCKX_FPSURF_H_
#define OSKAR_DIERCKX_FPSURF_H_

/**
 * @file oskar_dierckx_fpsurf.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Internal routine used by surfit.
 */
void oskar_dierckx_fpsurf(int iopt, int m, double *x, double *y,
        const double *z, const double *w, double xb, double xe, double yb,
        double ye, int kxx, int kyy, double s, int nxest, int nyest,
        double eta, double tol, int maxit, int nmax, int nc, int *nx0,
        double *tx, int *ny0, double *ty, double *c, double *fp, double *fp0,
        double *fpint, double *coord, double *f, double *ff, double *a,
        double *q, double *bx, double *by, double *spx, double *spy, double *h,
        int *index, int *nummer, double *wrk, int lwrk, int *ier);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DIERCKX_FPSURF_H_ */
