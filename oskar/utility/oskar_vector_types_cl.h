struct Real4c { Real2 a, b, c, d; } __attribute__((aligned(4*sizeof(Real2))));
typedef struct Real4c Real4c;
