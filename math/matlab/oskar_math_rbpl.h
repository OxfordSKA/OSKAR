#ifndef OSKAR_MATH_RBPL_H_
#define OSKAR_MATH_RBPL_H_

#ifdef __cplusplus
extern "C"
#endif
void oskar_rbpl(int n, double min, double max, double threshold, double power1,
        double power2, int seed, double * values);

#endif // OSKAR_MATH_RBPL_H_
