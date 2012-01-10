#ifndef DIERCKX_FPRATI_H_
#define DIERCKX_FPRATI_H_

/**
 * @file fprati.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Given three points (p1,f1),(p2,f2) and (p3,f3), function fprati
 * gives the value of p such that the rational interpolating function
 * of the form r(p) = (u*p+v)/(p+w) equals zero at p.
 */
float fprati_f(float *p1, float *f1, float p2, float f2, float *p3,
        float *f3);

/**
 * @details
 * Given three points (p1,f1),(p2,f2) and (p3,f3), function fprati
 * gives the value of p such that the rational interpolating function
 * of the form r(p) = (u*p+v)/(p+w) equals zero at p.
 */
double fprati_d(double *p1, double *f1, double p2, double f2, double *p3,
        double *f3);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPRATI_H_ */
