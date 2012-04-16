#ifndef DIERCKX_FPROTA_H_
#define DIERCKX_FPROTA_H_

/**
 * @file dierckx_fprota.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fprota applies a givens rotation to a and b.
 */
void dierckx_fprota_f(float cos_, float sin_, float *a, float *b);

/**
 * @details
 * Subroutine fprota applies a givens rotation to a and b.
 */
void dierckx_fprota_d(double cos_, double sin_, double *a, double *b);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPROTA_H_ */
