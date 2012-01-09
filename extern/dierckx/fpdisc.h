#ifndef DIERCKX_FPDISC_H_
#define DIERCKX_FPDISC_H_

/**
 * @file fpdisc.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fpdisc calculates the discontinuity jumps of the kth
 * derivative of the b-splines of degree k at the knots t(k+2)..t(n-k-1)
 */
void fpdisc(const float *t, int n, int k2, float *b, int nest);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPDISC_H_ */
