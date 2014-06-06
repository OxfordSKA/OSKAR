#ifndef OSKAR_DIERCKX_FPRANK_H_
#define OSKAR_DIERCKX_FPRANK_H_

/**
 * @file oskar_dierckx_fprank.h
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
void oskar_dierckx_fprank(double *a, double *f, int n, int m, int na,
        double tol, double *c, double *sq, int *rank, double *aa,
        double *ff, double *h);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DIERCKX_FPRANK_H_ */
