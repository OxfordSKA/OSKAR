#ifndef DIERCKX_FPBISP_H_
#define DIERCKX_FPBISP_H_

/**
 * @file fpbisp.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Internal routine used by bispev.
 */
void fpbisp(const float *tx, int nx, const float *ty, int ny,
    const float *c, int kx, int ky, const float *x, int mx, const float *y,
    int my, float *z, float *wx, float *wy, int *lx, int *ly);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPBISP_H_ */
