kernel void mem_scale_REAL(const int n, const REAL val, global REAL* a)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    a[i] *= val;
}
