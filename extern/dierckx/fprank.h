#ifndef DIERCKX_FPRANK_H_
#define DIERCKX_FPRANK_H_

/**
 * @file fprank.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fprank finds the minimum norm solution of a least-
 * squares problem in case of rank deficiency.
 *
 *  input parameters:
 *    a : array, which contains the non-zero elements of the observation
 *        matrix after triangularization by givens transformations.
 *    f : array, which contains the transformed right hand side.
 *    n : integer, which contains the dimension of a.
 *    m : integer, which denotes the bandwidth of a.
 *  tol : real value, giving a threshold to determine the rank of a.
 *
 *  output parameters:
 *    c : array, which contains the minimum norm solution.
 *   sq : real value, giving the contribution of reducing the rank
 *       to the sum of squared residuals.
 * rank : integer, which contains the rank of matrix a.
 */
void fprank(float *a, float *f, int n, int m, int na,
        float tol, float *c, float *sq, int *rank, float *aa,
        float *ff, float *h);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPRANK_H_ */
