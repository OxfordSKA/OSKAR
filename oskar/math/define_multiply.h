/* Copyright (c) 2013-2019, The University of Oxford. See LICENSE file. */

/* OUT_S = A * B */
#define OSKAR_MUL_COMPLEX(OUT_S, A, B) {\
        OUT_S.x = A.x * B.x - A.y * B.y;\
        OUT_S.y = A.x * B.y + A.y * B.x;}\

/* OUT_S = A * conj(B) */
#define OSKAR_MUL_COMPLEX_CONJUGATE(OUT_S, A, B) {\
        OUT_S.x = A.x * B.x + A.y * B.y;\
        OUT_S.y = A.y * B.x - A.x * B.y;}\

/* A *= B */
#define OSKAR_MUL_COMPLEX_IN_PLACE(FP2, A, B) {\
        const FP2 a1__ = A;\
        A.x *= B.x;         A.x -= a1__.y * B.y;\
        A.y = a1__.x * B.y; A.y += a1__.y * B.x;\
    }\

/* A *= conj(B) */
#define OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, A, B) {\
        const FP2 a1__ = A;\
        A.x *= B.x;         A.x += a1__.y * B.y;\
        A.y = a1__.y * B.x; A.y -= a1__.x * B.y;}\

/* M *= A */
#define OSKAR_MUL_COMPLEX_MATRIX_COMPLEX_SCALAR_IN_PLACE(FP2, M, A) {\
        OSKAR_MUL_COMPLEX_IN_PLACE(FP2, M.a, A)\
        OSKAR_MUL_COMPLEX_IN_PLACE(FP2, M.b, A)\
        OSKAR_MUL_COMPLEX_IN_PLACE(FP2, M.c, A)\
        OSKAR_MUL_COMPLEX_IN_PLACE(FP2, M.d, A)}\

/* M1 = M1 * M2
 * a = a1 a2 + b1 c2
 * b = a1 b2 + b1 d2
 * c = c1 a2 + d1 c2
 * d = c1 b2 + d1 d2 */
#define OSKAR_MUL_COMPLEX_MATRIX_IN_PLACE(FP2, M1, M2) {\
        FP2 t__; const FP2 a__ = M1.a; const FP2 c__ = M1.c;\
        OSKAR_MUL_COMPLEX_IN_PLACE(FP2, M1.a, M2.a)\
        OSKAR_MUL_COMPLEX(t__, M1.b, M2.c)\
        M1.a.x += t__.x; M1.a.y += t__.y;\
        OSKAR_MUL_COMPLEX_IN_PLACE(FP2, M1.c, M2.a)\
        OSKAR_MUL_COMPLEX(t__, M1.d, M2.c)\
        M1.c.x += t__.x; M1.c.y += t__.y;\
        OSKAR_MUL_COMPLEX_IN_PLACE(FP2, M1.b, M2.d)\
        OSKAR_MUL_COMPLEX(t__, a__, M2.b)\
        M1.b.x += t__.x; M1.b.y += t__.y;\
        OSKAR_MUL_COMPLEX_IN_PLACE(FP2, M1.d, M2.d)\
        OSKAR_MUL_COMPLEX(t__, c__, M2.b)\
        M1.d.x += t__.x; M1.d.y += t__.y;}\

/* M1 = M1 * conj_trans(M2) */
#define OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, M1, M2) {\
        FP2 t__; const FP2 a__ = M1.a; const FP2 c__ = M1.c;\
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, M1.a, M2.a)\
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, M1.b, M2.b)\
        M1.a.x += t__.x; M1.a.y += t__.y;\
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, M1.c, M2.a)\
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, M1.d, M2.b)\
        M1.c.x += t__.x; M1.c.y += t__.y;\
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, M1.b, M2.d)\
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, a__, M2.c)\
        M1.b.x += t__.x; M1.b.y += t__.y;\
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, M1.d, M2.d)\
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, c__, M2.c)\
        M1.d.x += t__.x; M1.d.y += t__.y;}\

/* M1 = M1 * M2
 * The second matrix must have a and d both real, with the form:
 *   ( a   b )
 *   ( -   d )
 */
#define OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, M1, M2) {\
        FP2 t__; const FP2 a__ = M1.a; const FP2 c__ = M1.c;\
        M1.a.x *= M2.a.x; M1.a.y *= M2.a.x;\
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, M1.b, M2.b)\
        M1.a.x += t__.x;  M1.a.y += t__.y;\
        M1.c.x *= M2.a.x; M1.c.y *= M2.a.x;\
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, M1.d, M2.b)\
        M1.c.x += t__.x;  M1.c.y += t__.y;\
        M1.b.x *= M2.d.x; M1.b.y *= M2.d.x;\
        OSKAR_MUL_COMPLEX(t__, a__, M2.b)\
        M1.b.x += t__.x;  M1.b.y += t__.y;\
        M1.d.x *= M2.d.x; M1.d.y *= M2.d.x;\
        OSKAR_MUL_COMPLEX(t__, c__, M2.b)\
        M1.d.x += t__.x;  M1.d.y += t__.y;}\

/* OUT_S += A * B */
#define OSKAR_MUL_ADD_COMPLEX(OUT_S, A, B) {\
        OUT_S.x += A.x * B.x; OUT_S.x -= A.y * B.y;\
        OUT_S.y += A.x * B.y; OUT_S.y += A.y * B.x;}\

/* OUT_S -= A * B */
#define OSKAR_MUL_SUB_COMPLEX(OUT_S, A, B) {\
        OUT_S.x -= A.x * B.x; OUT_S.x += A.y * B.y;\
        OUT_S.y -= A.x * B.y; OUT_S.y -= A.y * B.x;}\

/* OUT_S += A * conj(B) */
#define OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_S, A, B) {\
        OUT_S.x += A.x * B.x; OUT_S.x += A.y * B.y;\
        OUT_S.y += A.y * B.x; OUT_S.y -= A.x * B.y;}\

/* OUT_M = M * A */
#define OSKAR_MUL_COMPLEX_MATRIX_COMPLEX_SCALAR(OUT_M, M, A) {\
        OSKAR_MUL_COMPLEX(OUT_M.a, M.a, A)\
        OSKAR_MUL_COMPLEX(OUT_M.b, M.b, A)\
        OSKAR_MUL_COMPLEX(OUT_M.c, M.c, A)\
        OSKAR_MUL_COMPLEX(OUT_M.d, M.d, A)}\

/* OUT_M += M1 * A */
#define OSKAR_MUL_ADD_COMPLEX_MATRIX_SCALAR(OUT_M, M1, A) {\
        OUT_M.a.x += M1.a.x * A; OUT_M.a.y += M1.a.y * A;\
        OUT_M.b.x += M1.b.x * A; OUT_M.b.y += M1.b.y * A;\
        OUT_M.c.x += M1.c.x * A; OUT_M.c.y += M1.c.y * A;\
        OUT_M.d.x += M1.d.x * A; OUT_M.d.y += M1.d.y * A;}\

/* OUT_M = M1 * M2
 * a = a1 a2 + b1 c2
 * b = a1 b2 + b1 d2
 * c = c1 a2 + d1 c2
 * d = c1 b2 + d1 d2 */
#define OSKAR_MUL_COMPLEX_MATRIX(OUT_M, M1, M2) {\
        OSKAR_MUL_COMPLEX(OUT_M.a, M1.a, M2.a)\
        OSKAR_MUL_COMPLEX(OUT_M.b, M1.a, M2.b)\
        OSKAR_MUL_COMPLEX(OUT_M.c, M1.c, M2.a)\
        OSKAR_MUL_COMPLEX(OUT_M.d, M1.c, M2.b)\
        OSKAR_MUL_ADD_COMPLEX(OUT_M.a, M1.b, M2.c)\
        OSKAR_MUL_ADD_COMPLEX(OUT_M.b, M1.b, M2.d)\
        OSKAR_MUL_ADD_COMPLEX(OUT_M.c, M1.d, M2.c)\
        OSKAR_MUL_ADD_COMPLEX(OUT_M.d, M1.d, M2.d)}\

/* OUT_M = M1 * conj_trans(M2)
 * a = a1 a2* + b1 b2*
 * b = a1 c2* + b1 d2*
 * c = c1 a2* + d1 b2*
 * d = c1 c2* + d1 d2* */
#define OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE(OUT_M, M1, M2) {\
        OSKAR_MUL_COMPLEX_CONJUGATE(OUT_M.a, M1.a, M2.a)\
        OSKAR_MUL_COMPLEX_CONJUGATE(OUT_M.b, M1.a, M2.c)\
        OSKAR_MUL_COMPLEX_CONJUGATE(OUT_M.c, M1.c, M2.a)\
        OSKAR_MUL_COMPLEX_CONJUGATE(OUT_M.d, M1.c, M2.c)\
        OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_M.a, M1.b, M2.b)\
        OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_M.b, M1.b, M2.d)\
        OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_M.c, M1.d, M2.b)\
        OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_M.d, M1.d, M2.d)}\

/* Clears a complex matrix. */
#define OSKAR_CLEAR_COMPLEX_MATRIX(FP, M) {\
        MAKE_ZERO2(FP, M.a); MAKE_ZERO2(FP, M.b);\
        MAKE_ZERO2(FP, M.c); MAKE_ZERO2(FP, M.d);}\

/* OUT_V = M * IN */
#define OSKAR_MUL_3X3_MATRIX_VECTOR(OUT_V, M, V) {\
        OUT_V[0] = M[0] * V[0] + M[1] * V[1] + M[2] * V[2];\
        OUT_V[1] = M[3] * V[0] + M[4] * V[1] + M[5] * V[2];\
        OUT_V[2] = M[6] * V[0] + M[7] * V[1] + M[8] * V[2];}\

