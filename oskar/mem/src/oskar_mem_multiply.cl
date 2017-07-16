#define C_MUL(A, B, OUT) \
    OUT.x = A.x * B.x; OUT.x -= A.y * B.y; \
    OUT.y = A.x * B.y; OUT.y += A.y * B.x;

kernel void mem_multiply_rr_r_REAL(const int n,
        global const REAL* a, global const REAL* b, global REAL* c)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    c[i] = a[i] * b[i];
}

kernel void mem_multiply_cc_c_REAL(const int n,
        global const REAL2* a, global const REAL2* b, global REAL2* c)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    REAL2 a_, b_, t;
    a_ = a[i];
    b_ = b[i];
    C_MUL(a_, b_, t)
    c[i] = t;
}

kernel void mem_multiply_cc_m_REAL(const int n,
        global const REAL2* a, global const REAL2* b, global REAL2* c)
{
    const int i = get_global_id(0);
    const int j = 4 * i;
    if (i >= n) return;
    REAL2 a_, b_, t;
    a_ = a[i];
    b_ = b[i];
    C_MUL(a_, b_, t)
    c[j]     = t;
    c[j + 1] = (REAL2)(0.0, 0.0);
    c[j + 2] = (REAL2)(0.0, 0.0);
    c[j + 3] = t;
}

kernel void mem_multiply_cm_m_REAL(const int n,
        global const REAL2* a, global const REAL2* b, global REAL2* c)
{
    const int i = get_global_id(0);
    const int j = 4 * i;
    if (i >= n) return;
    REAL2 a_, b0, b1, b2, b3, t;
    a_ = a[i];
    b0 = b[j];
    b1 = b[j + 1];
    b2 = b[j + 2];
    b3 = b[j + 3];
    C_MUL(a_, b0, t)
    b0 = t;
    C_MUL(a_, b1, t)
    b1 = t;
    C_MUL(a_, b2, t)
    b2 = t;
    C_MUL(a_, b3, t)
    c[j]     = b0;
    c[j + 1] = b1;
    c[j + 2] = b2;
    c[j + 3] = t;
}

kernel void mem_multiply_mc_m_REAL(const int n,
        global const REAL2* a, global const REAL2* b, global REAL2* c)
{
    const int i = get_global_id(0);
    const int j = 4 * i;
    if (i >= n) return;
    REAL2 a0, a1, a2, a3, b_, t;
    a0 = a[j];
    a1 = a[j + 1];
    a2 = a[j + 2];
    a3 = a[j + 3];
    b_ = b[i];
    C_MUL(a0, b_, t)
    a0 = t;
    C_MUL(a1, b_, t)
    a1 = t;
    C_MUL(a2, b_, t)
    a2 = t;
    C_MUL(a3, b_, t)
    c[j]     = a0;
    c[j + 1] = a1;
    c[j + 2] = a2;
    c[j + 3] = t;
}

kernel void mem_multiply_mm_m_REAL(const int n,
        global const REAL2* a, global const REAL2* b, global REAL2* c)
{
    const int i = get_global_id(0);
    const int j = 4 * i;
    if (i >= n) return;
    REAL2 a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, t1, t2;
    a0 = a[j];
    a1 = a[j + 1];
    a2 = a[j + 2];
    a3 = a[j + 3];
    b0 = b[j];
    b1 = b[j + 1];
    b2 = b[j + 2];
    b3 = b[j + 3];
    C_MUL(a0, b0, t1)
    C_MUL(a1, b2, t2)
    c0 = t1 + t2;
    C_MUL(a0, b1, t1)
    C_MUL(a1, b3, t2)
    c1 = t1 + t2;
    C_MUL(a2, b0, t1)
    C_MUL(a3, b2, t2)
    c2 = t1 + t2;
    C_MUL(a2, b1, t1)
    C_MUL(a3, b3, t2)
    c3 = t1 + t2;
    c[j]     = c0;
    c[j + 1] = c1;
    c[j + 2] = c2;
    c[j + 3] = c3;
}

#undef C_MUL
