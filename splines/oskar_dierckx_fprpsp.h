#ifndef OSKAR_DIERCKX_FPRPSP_H_
#define OSKAR_DIERCKX_FPRPSP_H_

/**
 * @file oskar_dierckx_fprpsp.h
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
void oskar_dierckx_fprpsp(int nt, int np, const double *co, const double *si,
        double *c, double *f, int ncoff);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DIERCKX_FPRPSP_H_ */

