#ifndef DIERCKX_FPROTA_H_
#define DIERCKX_FPROTA_H_

/**
 * @file fprota.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fprota applies a givens rotation to a and b.
 */
void fprota(float cos_, float sin_, float *a, float *b);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPROTA_H_ */
