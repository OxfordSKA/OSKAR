#ifndef DIERCKX_FPRPSP_H_
#define DIERCKX_FPRPSP_H_

/**
 * @file fprpsp.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Given the coefficients of a spherical spline function, subroutine
 * fprpsp calculates the coefficients in the standard b-spline
 * representation of this bicubic spline.
 */
void fprpsp(int nt, int np, const float *co, const float *si,
        float *c, float *f, int ncoff);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPRPSP_H_ */
