#ifndef DIERCKX_FPBSPL_H_
#define DIERCKX_FPBSPL_H_

/**
 * @file fpbspl.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fpbspl evaluates the (k+1) non-zero b-splines of
 * degree k at t(l) <= x < t(l+1) using the stable recurrence
 * relation of de Boor and Cox.
 */
void fpbspl(const float *t, int k, float x, int l, float *h);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPBSPL_H_ */
