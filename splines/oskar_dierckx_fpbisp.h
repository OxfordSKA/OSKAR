#ifndef OSKAR_DIERCKX_FPBISP_H_
#define OSKAR_DIERCKX_FPBISP_H_

/**
 * @file oskar_dierckx_fpbisp.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Internal routine used by bispev.
 */
void oskar_dierckx_fpbisp_f(const float *tx, int nx, const float *ty, int ny,
    const float *c, int kx, int ky, const float *x, int mx, const float *y,
    int my, float *z, float *wx, float *wy, int *lx, int *ly);

/**
 * @details
 * Internal routine used by bispev.
 */
void oskar_dierckx_fpbisp_d(const double *tx, int nx, const double *ty, int ny,
    const double *c, int kx, int ky, const double *x, int mx, const double *y,
    int my, double *z, double *wx, double *wy, int *lx, int *ly);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DIERCKX_FPBISP_H_ */
