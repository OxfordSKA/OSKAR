kernel void mem_add_REAL(const int n,
        global const REAL* a, global const REAL* b, global REAL* c)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    c[i] = a[i] + b[i];
}
