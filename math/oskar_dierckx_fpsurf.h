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
void oskar_dierckx_fpsurf_f(int iopt, int m, float *x, float *y,
        const float *z, const float *w, float xb, float xe, float yb,
        float ye, int kxx, int kyy, float s, int nxest, int nyest,
        float eta, float tol, int maxit, int nmax, int nc, int *nx0,
        float *tx, int *ny0, float *ty, float *c, float *fp, float *fp0,
        float *fpint, float *coord, float *f, float *ff, float *a,
        float *q, float *bx, float *by, float *spx, float *spy, float *h,
        int *index, int *nummer, float *wrk, int lwrk, int *ier);

/**
 * @details
 * Internal routine used by surfit.
 */
void oskar_dierckx_fpsurf_d(int iopt, int m, double *x, double *y,
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
