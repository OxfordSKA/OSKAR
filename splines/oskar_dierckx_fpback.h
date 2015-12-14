#ifndef OSKAR_DIERCKX_FPBACK_H_
#define OSKAR_DIERCKX_FPBACK_H_

/**
 * @file oskar_dierckx_fpback.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fpback calculates the solution of the system of
 * equations a*c = z with a a n x n upper triangular matrix
 * of bandwidth k.
 */
void oskar_dierckx_fpback(const double *a, const double *z, int n, int k,
        double *c, int nest);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DIERCKX_FPBACK_H_ */
