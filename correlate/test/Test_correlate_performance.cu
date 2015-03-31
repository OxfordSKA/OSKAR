#include <gtest/gtest.h>

#include <oskar_global.h> // need this first to define __CUDACC__
#include <oskar_vector_types.h>
#include <oskar_cross_correlate_point_time_smearing_cuda.h>
#include <private_correlate_functions_inline.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <cmath>
#include <cstdio>

using namespace std;

#define OMEGA_EARTH  7.272205217e-5  /* radians/sec */
#define OMEGA_EARTHf 7.272205217e-5f /* radians/sec */

#define NUM_THREADS 128
#define BLOCK_SIZE 2 // Number of baselines in a block

typedef const float* __restrict__ Array;
typedef const float4c* __restrict__ MArray;

extern __shared__ float4c  smem_f4c[];

void correlate_f(int num_sources, int num_stations, const float4c* d_Jones,
        const float* d_I, const float* d_Q, const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n, const float* d_u,
        const float* d_v, const float* d_x, const float* d_y,
        float freq, float bandwidth, float time_int, float gha0, float dec0,
        float4c* d_vis);
__global__
void correlate_cudak_f(const int num_sources, const int num_stations,
        MArray d_Jones, Array d_I, Array d_Q, Array d_U,
        Array d_V, Array d_l, Array d_m, Array d_n, Array d_u, Array d_v,
        Array d_x, Array d_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* d_vis);
void correlate_warpshuffle_f(int num_sources, int num_stations, const float4c* d_Jones,
        const float* d_I, const float* d_Q, const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n, const float* d_u,
        const float* d_v, const float* d_x, const float* d_y,
        float freq, float bandwidth, float time_int, float gha0, float dec0,
        float4c* d_vis);
__global__ void
correlate_warpshuffle_cudak_f(const int num_sources, const int num_stations,
        MArray d_Jones, Array d_I, Array d_Q, Array d_U,
        Array d_V, Array d_l, Array d_m, Array d_n, Array d_u, Array d_v,
        Array d_x, Array d_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* d_vis);
void correlate_warpshuffle_blocked_f(int num_sources, int num_stations, const float4c* d_Jones,
        const float* d_I, const float* d_Q, const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n, const float* d_u,
        const float* d_v, const float* d_x, const float* d_y,
        float freq, float bandwidth, float time_int, float gha0, float dec0,
        float4c* d_vis);
__global__ void
correlate_warpshuffle_blocked_cudak_f(const int num_sources, const int num_stations,
        MArray d_Jones, Array d_I, Array d_Q, Array d_U,
        Array d_V, Array d_l, Array d_m, Array d_n, Array d_u, Array d_v,
        Array d_x, Array d_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* d_vis);
void correlate_warpshuffle_blocked_smem_f(int num_sources, int num_stations, const float4c* d_Jones,
        const float* d_I, const float* d_Q, const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n, const float* d_u,
        const float* d_v, const float* d_x, const float* d_y,
        float freq, float bandwidth, float time_int, float gha0, float dec0,
        float4c* d_vis);
__global__ void
correlate_warpshuffle_blocked_smem_cudak_f(const int num_sources, const int num_stations,
        MArray d_Jones, Array d_I, Array d_Q, Array d_U,
        Array d_V, Array d_l, Array d_m, Array d_n, Array d_u, Array d_v,
        Array d_x, Array d_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* d_vis);
//__global__ void test_warp_shuffle(float* value);
void print_float4c(float4c value);
void print_float4c_range(int i0, int i1, float4c* values);

//==============================================================================

//class InputData : public ::testing::Test
//{
//
//};

TEST(PerformanceTests, test01)
{
    // ----------- Allocate host memory ----------------------------------------
    int num_sources  = 128*100;
    int num_stations = 100;
    float freq       = 100.0e6; // [Hz]
    float bandwidth  = 1.0e5;   // [Hz]
    float time_int   = 1.0f;    // [sec]
    float gha0       = 0.0f;    // Greenwich hour angle of phase centre [radians]
    float dec0       = 0.0f;    // [radians]

    int num_baselines = (num_stations*(num_stations-1))/2;

    float *h_I, *h_Q, *h_U, *h_V, *h_l, *h_m, *h_n;
    size_t memSize = num_sources * sizeof(float);
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_I, memSize));
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_Q, memSize));
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_U, memSize));
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_V, memSize));
    //ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_l, memSize));
    h_l = (float*)malloc(memSize);
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_m, memSize));
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_n, memSize));

    memSize = num_stations * sizeof(float);
    float *h_u, *h_v, *h_x, *h_y;
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_u, memSize));
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_v, memSize));
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_x, memSize));
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_y, memSize));

    float4c* h_Jones;
    memSize = num_sources * num_stations * sizeof(float4c);
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_Jones, memSize));

    float4c* h_vis;
    memSize = num_baselines * sizeof(float4c);
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_vis, memSize));


    // ----------- Fill Host memory --------------------------------------------
    float rLO = -0.1;
    float rHI =  0.1;
    for (int i = 0; i < num_sources; ++i) {
        h_I[i] = 1.0f;
        h_Q[i] = 0.3f;
        h_U[i] = 0.2f;
        h_V[i] = 0.0f;
        h_l[i] = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
        h_m[i] = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
        h_n[i] = sqrt(h_l[i]*h_l[i] + h_m[i]*h_m[i]);
    }

    rLO = 0.0;
    rHI = 5.0;
    for (int i = 0; i < num_stations; ++i) {
        h_u[i] = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
        h_v[i] = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
        h_x[i] = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
        h_y[i] = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
    }

    rLO = 0.0;
    rHI = 0.1;
    for (int idx = 0, j = 0; j < num_stations; ++j) {
        for (int i = 0; i < num_sources; ++i,++idx) {
            float x = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
            float y = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
            h_Jones[idx].a = make_float2(x, y);
            x = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
            y = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
            h_Jones[idx].b = make_float2(x, y);
            x = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
            y = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
            h_Jones[idx].c = make_float2(x, y);
            x = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
            y = rLO + (float)rand()/((float)RAND_MAX/(rHI-rLO));
            h_Jones[idx].d = make_float2(x, y);
        }
    }

    //    print_float4c_range(0, 4, h_Jones);

    // ----------- Allocate device memory --------------------------------------
    float *d_I, *d_Q, *d_U, *d_V, *d_l, *d_m, *d_n;
    memSize = num_sources * sizeof(float);
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_I, memSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_Q, memSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_U, memSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_V, memSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_l, memSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_m, memSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_n, memSize));

    float *d_u, *d_v, *d_x, *d_y;
    memSize = num_stations * sizeof(float);
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_u, memSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_v, memSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_x, memSize));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_y, memSize));

    float4c* d_Jones;
    memSize = num_sources * num_stations * sizeof(float4c);
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_Jones, memSize));

    float4c* d_vis;
    memSize = num_baselines * sizeof(float4c);
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_vis, memSize));


    // ----------- Copy host -> device -----------------------------------------
    cudaMemcpyKind kind = cudaMemcpyHostToDevice;
    memSize = num_sources * sizeof(float);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_I, h_I, memSize, kind));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_Q, h_Q, memSize, kind));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_U, h_U, memSize, kind));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_V, h_V, memSize, kind));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_l, h_l, memSize, kind));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_m, h_m, memSize, kind));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_n, h_n, memSize, kind));

    memSize = num_stations * sizeof(float);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_u, h_u, memSize, kind));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_v, h_v, memSize, kind));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_x, h_x, memSize, kind));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_y, h_y, memSize, kind));

    memSize = num_stations * num_sources * sizeof(float4c);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_Jones, h_Jones, memSize, kind));


    // ----------- 'Correlate' -------------------------------------------------
    cudaMemset(d_vis, 0.0, num_baselines * sizeof(float4c));
    correlate_f(num_sources, num_stations, d_Jones, d_I, d_Q, d_U, d_V, d_l,
            d_m, d_n, d_u, d_v, d_x, d_y, freq, bandwidth, time_int, gha0,
            dec0, d_vis);

    // ----------- copy back visibilities --------------------------------------
    kind = cudaMemcpyDeviceToHost;
    memSize = num_baselines * sizeof(float4c);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_vis, d_vis, memSize, kind));
    print_float4c_range(0, 5, h_vis);


    // ----------- 'Correlate' -------------------------------------------------
    cudaMemset(d_vis, 0.0, num_baselines * sizeof(float4c));
    correlate_warpshuffle_f(num_sources, num_stations, d_Jones, d_I, d_Q, d_U, d_V, d_l,
            d_m, d_n, d_u, d_v, d_x, d_y, freq, bandwidth, time_int, gha0,
            dec0, d_vis);

    // ----------- copy back visibilities --------------------------------------
    kind = cudaMemcpyDeviceToHost;
    memSize = num_baselines * sizeof(float4c);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_vis, d_vis, memSize, kind));
    printf("\n");
    print_float4c_range(0, 5, h_vis);

//    // ----------- 'Correlate' -------------------------------------------------
//    cudaMemset(d_vis, 0.0, num_baselines * sizeof(float4c));
//    correlate_warpshuffle_blocked_smem_f(num_sources, num_stations, d_Jones, d_I, d_Q, d_U, d_V, d_l,
//            d_m, d_n, d_u, d_v, d_x, d_y, freq, bandwidth, time_int, gha0,
//            dec0, d_vis);
//
//    // ----------- copy back visibilities --------------------------------------
//    kind = cudaMemcpyDeviceToHost;
//    memSize = num_baselines * sizeof(float4c);
//    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_vis, d_vis, memSize, kind));
//    printf("\n");
//    print_float4c_range(0, 5, h_vis);

    //    // ----------- 'Correlate' -------------------------------------------------
    //    cudaMemset(d_vis, 0.0, num_baselines * sizeof(float4c));
    //    correlate_warpshuffle_blocked_f(num_sources, num_stations, d_Jones, d_I, d_Q, d_U, d_V, d_l,
    //            d_m, d_n, d_u, d_v, d_x, d_y, freq, bandwidth, time_int, gha0,
    //            dec0, d_vis);
    //
    //    // ----------- copy back visibilities --------------------------------------
    //    kind = cudaMemcpyDeviceToHost;
    //    memSize = num_baselines * sizeof(float4c);
    //    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_vis, d_vis, memSize, kind));
    //    printf("\n");
    //    print_float4c_range(0, 5, h_vis);


    //    float* d_warp_sum;
    //    cudaMalloc(&d_warp_sum, sizeof(float));
    //    cudaMemset(d_warp_sum, 3, sizeof(float));
    //    dim3 blockDim(32,1,1);
    //    dim3 gridDim(1,1,1);
    //    test_warp_shuffle <<< gridDim, blockDim >>> (d_warp_sum);
    //    float* h_warp_sum;
    //    cudaMallocHost(&h_warp_sum, sizeof(float));
    //    cudaMemcpy(h_warp_sum, d_warp_sum, sizeof(float), cudaMemcpyDeviceToHost);
    //    printf("warp sum = %f\n", h_warp_sum[0]);
}


void correlate_f(int num_sources, int num_stations, const float4c* d_Jones,
        const float* d_I, const float* d_Q, const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n, const float* d_u,
        const float* d_v, const float* d_x, const float* d_y,
        float freq, float bandwidth, float time_int, float gha0, float dec0,
        float4c* d_vis)
{
    dim3 num_threads(NUM_THREADS, 1);
    dim3 num_blocks(num_stations, num_stations);
    size_t shared_mem = num_threads.x * sizeof(float4c);
    //cudaFuncSetCacheConfig(correlate_cudak_f, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(correlate_cudak_f, cudaFuncCachePreferShared);
    correlate_cudak_f OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_Jones,
            d_I, d_Q, d_U, d_V, d_l, d_m, d_n, d_u, d_v, d_x, d_y,
            freq, bandwidth, time_int, gha0, dec0, d_vis);
}

__global__ void
//__launch_bounds__(128, 8)
correlate_cudak_f(const int num_sources, const int num_stations,
        MArray d_Jones, Array d_I, Array d_Q, Array d_U,
        Array d_V, Array d_l, Array d_m, Array d_n, Array d_u, Array d_v,
        Array d_x, Array d_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* d_vis)
        {
    /* Common values per thread block. */
    __shared__ float uu, vv, du_dt, dv_dt, dw_dt;
    __shared__ const float4c* __restrict__ station_i;
    __shared__ const float4c* __restrict__ station_j;

    int SI = blockIdx.x;
    int SJ = blockIdx.y;

    /* Return immediately if in the wrong half of the visibility matrix. */
    if (SJ >= SI) return;

    /* Use thread 0 to set up the block. */
    if (threadIdx.x == 0)
    {
        /* Baseline lengths. */
        uu = (d_u[SI] - d_u[SJ]) * 0.5f;
        vv = (d_v[SI] - d_v[SJ]) * 0.5f;

        /* Modify the baseline distance to include the common components
         * of the bandwidth smearing term. */
        float temp = bandwidth_hz / freq_hz; /* Fractional bandwidth */
        uu *= temp;
        vv *= temp;

        /* Compute the derivatives for time-average smearing. */
        float sin_HA, cos_HA, sin_Dec, cos_Dec;
        sincosf(gha0_rad, &sin_HA, &cos_HA);
        sincosf(dec0_rad, &sin_Dec, &cos_Dec);
        float xx = (d_x[SI] - d_x[SJ]) * 0.5f;
        float yy = (d_y[SI] - d_y[SJ]) * 0.5f;
        float rot_angle = OMEGA_EARTHf * time_int_sec;
        temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
        du_dt = (xx * cos_HA - yy * sin_HA) * rot_angle;
        dv_dt = temp * sin_Dec;
        dw_dt = -temp * cos_Dec;

        /* Get pointers to source vectors for both stations. */
        station_i = &d_Jones[num_sources * SI];
        station_j = &d_Jones[num_sources * SJ];
    }
    __syncthreads();

    /* Partial sum per thread. */
    float4c sum;
    sum.a = make_float2(0.0f, 0.0f);
    sum.b = sum.a;
    sum.c = sum.a;
    sum.d = sum.a;

    /* Each thread loops over a subset of the sources. */
    for (int i = threadIdx.x; i < num_sources; i += blockDim.x)
    {
        /* Get source direction cosines. */
        float l = d_l[i];
        float m = d_m[i];
        float n = d_n[i];

        /* Compute bandwidth- and time-smearing terms. */
        float a = uu * l + vv * m;
        float rb = oskar_sinc_f(a);
        float rt = oskar_sinc_f(du_dt * l + dv_dt * m + dw_dt * n);
        rb *= rt;

        /* Accumulate baseline visibility response for source. */
        oskar_accumulate_baseline_visibility_for_source_inline_f(&sum, i,
                d_I, d_Q, d_U, d_V, station_i, station_j, rb);
    }

    /* Store partial sum for the thread in shared memory and synchronise. */
    smem_f4c[threadIdx.x] = sum;
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources for this baseline. */
        sum.a = make_float2(0.0f, 0.0f);
        sum.b = sum.a;
        sum.c = sum.a;
        sum.d = sum.a;
        for (int i = 0; i < blockDim.x; ++i)
        {
            sum.a.x += smem_f4c[i].a.x;
            sum.a.y += smem_f4c[i].a.y;
            sum.b.x += smem_f4c[i].b.x;
            sum.b.y += smem_f4c[i].b.y;
            sum.c.x += smem_f4c[i].c.x;
            sum.c.y += smem_f4c[i].c.y;
            sum.d.x += smem_f4c[i].d.x;
            sum.d.y += smem_f4c[i].d.y;
        }

        /* Determine 1D visibility index for global memory store. */
        int i = SJ*(num_stations-1) - (SJ-1)*SJ/2 + SI - SJ - 1;

        /* Add result of this thread block to the baseline visibility. */
        d_vis[i].a.x += sum.a.x;
        d_vis[i].a.y += sum.a.y;
        d_vis[i].b.x += sum.b.x;
        d_vis[i].b.y += sum.b.y;
        d_vis[i].c.x += sum.c.x;
        d_vis[i].c.y += sum.c.y;
        d_vis[i].d.x += sum.d.x;
        d_vis[i].d.y += sum.d.y;
    }
}


void correlate_warpshuffle_f(int num_sources, int num_stations, const float4c* d_Jones,
        const float* d_I, const float* d_Q, const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n, const float* d_u,
        const float* d_v, const float* d_x, const float* d_y,
        float freq, float bandwidth, float time_int, float gha0, float dec0,
        float4c* d_vis)
{
    dim3 num_threads(NUM_THREADS, 1);
    dim3 num_blocks(num_stations-1, num_stations-1);
    int num_warps = num_threads.x / 32;
    size_t shared_mem = num_warps * sizeof(float4c);
    cudaFuncSetCacheConfig(correlate_cudak_f, cudaFuncCachePreferShared);
    correlate_warpshuffle_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_Jones,
            d_I, d_Q, d_U, d_V, d_l, d_m, d_n, d_u, d_v, d_x, d_y,
            freq, bandwidth, time_int, gha0, dec0, d_vis);
}

// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__global__ void
//__launch_bounds__(256, 8)
correlate_warpshuffle_cudak_f(const int num_sources, const int num_stations,
        MArray d_Jones, Array d_I, Array d_Q, Array d_U,
        Array d_V, Array d_l, Array d_m, Array d_n, Array d_u, Array d_v,
        Array d_x, Array d_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* d_vis)
{
#if __CUDA_ARCH__ >= 300
    /* Common values per thread block. */
    __shared__ float uu, vv, du_dt, dv_dt, dw_dt;
    __shared__ const float4c* __restrict__ station_i;
    __shared__ const float4c* __restrict__ station_j;

    int SI = blockIdx.x;
    int SJ = blockIdx.y;

    /* Return immediately if in the wrong half of the visibility matrix. */
    if (SJ >= SI) return;

    /* Use thread 0 to set up the block. */
    if (threadIdx.x == 0)
    {
        /* Baseline lengths. */
        uu = (d_u[SI] - d_u[SJ]) * 0.5f;
        vv = (d_v[SI] - d_v[SJ]) * 0.5f;

        /* Modify the baseline distance to include the common components
         * of the bandwidth smearing term. */
        float temp = channel_bandwidth_hz / freq_hz; /* Fractional bandwidth */
        uu *= temp;
        vv *= temp;

        /* Compute the derivatives for time-average smearing. */
        float sin_HA, cos_HA, sin_Dec, cos_Dec;
        sincosf(gha0_rad, &sin_HA, &cos_HA);
        sincosf(dec0_rad, &sin_Dec, &cos_Dec);
        float xx = (d_x[SI] - d_x[SJ]) * 0.5f;
        float yy = (d_y[SI] - d_y[SJ]) * 0.5f;
        float rot_angle = OMEGA_EARTHf * time_average_sec;
        temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
        du_dt = (xx * cos_HA - yy * sin_HA) * rot_angle;
        dv_dt = temp * sin_Dec;
        dw_dt = -temp * cos_Dec;

        /* Get pointers to source vectors for both stations. */
        station_i = &d_Jones[num_sources * SI];
        station_j = &d_Jones[num_sources * SJ];
    }
    __syncthreads();

    /* Partial sum per thread. */
    float4c sum;
    sum.a = make_float2(0.0f, 0.0f);
    sum.b = sum.a;
    sum.c = sum.a;
    sum.d = sum.a;

    /* Each thread loops over a subset of the sources. */
    for (int i = threadIdx.x; i < num_sources; i += blockDim.x)
    {
        /* Get source direction cosines. */
        float l = d_l[i];
        float m = d_m[i];
        float n = d_n[i];

        /* Compute bandwidth- and time-smearing terms. */
        float a = uu * l + vv * m;
        float rb = oskar_sinc_f(a);
        float rt = oskar_sinc_f(du_dt * l + dv_dt * m + dw_dt * n);
        rb *= rt;

        /* Accumulate baseline visibility response for source. */
        oskar_accumulate_baseline_visibility_for_source_inline_f(&sum, i,
                d_I, d_Q, d_U, d_V, station_i, station_j, rb);
    }

    // Warp shuffle reduction
    int laneID = threadIdx.x & 0x1f;
    for (int i = warpSize/2; i >=1; i/=2) {
        sum.a.x += __shfl_xor(sum.a.x, i, warpSize);
        sum.a.y += __shfl_xor(sum.a.y, i, warpSize);
        sum.b.x += __shfl_xor(sum.b.x, i, warpSize);
        sum.b.y += __shfl_xor(sum.b.y, i, warpSize);
        sum.c.x += __shfl_xor(sum.c.x, i, warpSize);
        sum.c.y += __shfl_xor(sum.c.y, i, warpSize);
        sum.d.x += __shfl_xor(sum.d.x, i, warpSize);
        sum.d.y += __shfl_xor(sum.d.y, i, warpSize);
    }
    // Store the partial sum (per warp) into shared memory.
    if (laneID == 0) {
        int warp = threadIdx.x / warpSize;
        smem_f4c[warp] = sum;
    }

    // Make sure all warps have finished.
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources for this baseline. */
        sum.a = make_float2(0.0f, 0.0f);
        sum.b = make_float2(0.0f, 0.0f);
        sum.c = make_float2(0.0f, 0.0f);
        sum.d = make_float2(0.0f, 0.0f);
        int num_warps = blockDim.x/warpSize;
        for (int i = 0; i < num_warps; ++i)
        {
            sum.a.x += smem_f4c[i].a.x;
            sum.a.y += smem_f4c[i].a.y;
            sum.b.x += smem_f4c[i].b.x;
            sum.b.y += smem_f4c[i].b.y;
            sum.c.x += smem_f4c[i].c.x;
            sum.c.y += smem_f4c[i].c.y;
            sum.d.x += smem_f4c[i].d.x;
            sum.d.y += smem_f4c[i].d.y;
        }

        /* Determine 1D visibility index for global memory store. */
        int i = SJ*(num_stations-1) - (SJ-1)*SJ/2 + SI - SJ - 1;

        /* Add result of this thread block to the baseline visibility. */
        d_vis[i].a.x += sum.a.x;
        d_vis[i].a.y += sum.a.y;
        d_vis[i].b.x += sum.b.x;
        d_vis[i].b.y += sum.b.y;
        d_vis[i].c.x += sum.c.x;
        d_vis[i].c.y += sum.c.y;
        d_vis[i].d.x += sum.d.x;
        d_vis[i].d.y += sum.d.y;
    }
#endif /* __CUDA_ARCH__ >= 300 */
}








void correlate_warpshuffle_blocked_smem_f(int num_sources, int num_stations, const float4c* d_Jones,
        const float* d_I, const float* d_Q, const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n, const float* d_u,
        const float* d_v, const float* d_x, const float* d_y,
        float freq, float bandwidth, float time_int, float gha0, float dec0,
        float4c* d_vis)
{
    int num_blocks = ((num_stations-1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(num_blocks, num_stations-1);
    dim3 blockDim(NUM_THREADS, 1);
    int num_warps = blockDim.x / 32;
    size_t shared_mem = (num_warps * BLOCK_SIZE) * sizeof(float4c);
    cudaFuncSetCacheConfig(correlate_cudak_f, cudaFuncCachePreferShared);
    correlate_warpshuffle_blocked_smem_cudak_f
    OSKAR_CUDAK_CONF(gridDim, blockDim, shared_mem)
    (num_sources, num_stations, d_Jones,
            d_I, d_Q, d_U, d_V, d_l, d_m, d_n, d_u, d_v, d_x, d_y,
            freq, bandwidth, time_int, gha0, dec0, d_vis);
}

// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__global__ void
//__launch_bounds__(128, 16)
correlate_warpshuffle_blocked_smem_cudak_f(const int num_sources, const int num_stations,
        MArray d_Jones, Array d_I, Array d_Q, Array d_U,
        Array d_V, Array d_l, Array d_m, Array d_n, Array d_u, Array d_v,
        Array d_x, Array d_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* d_vis)
{
#if __CUDA_ARCH__ >= 300
    // This kernel evaluates baselines p-q for a single station p and a range
    // of stations q, determined by the BLOCK_SIZE

    // Index of first station q in the block.
    int q0 = (blockIdx.y+1) + (blockIdx.x*BLOCK_SIZE);

    // Return immediately if the block is outside range of valid baselines.
    if (q0 > num_stations)
        return;

    // Station index p (common to all baselines processed by this block)
    int p = blockIdx.y;

    // TODO fix this for the case where blockDim doesn't divide perfectly by
    // warp size
    int num_warps = blockDim.x/warpSize;

    /* Cache common (per baseline values). */
    __shared__ float uu[BLOCK_SIZE], vv[BLOCK_SIZE];
    __shared__ float du_dt[BLOCK_SIZE], dv_dt[BLOCK_SIZE], dw_dt[BLOCK_SIZE];
    __shared__ MArray station_p;

    if (threadIdx.x == 0)
        station_p = &d_Jones[num_sources * p];

    /* Use the first BLOCK_SIZE threads to cache the per baseline values:
     * uu, vv, du_dt, dv_dt, & dw_dt */
    if (threadIdx.x < BLOCK_SIZE)
    {
        int q = q0 + threadIdx.x; // index of station q in the baseline

        /* Baseline lengths. */
        uu[threadIdx.x] = (d_u[q] - d_u[p]) * 0.5f;
        vv[threadIdx.x] = (d_v[q] - d_v[p]) * 0.5f;

        /* Modify the baseline distance to include the common components
         * of the bandwidth smearing term. */
        float temp = channel_bandwidth_hz / freq_hz; /* Fractional bandwidth */
        uu[threadIdx.x] *= temp;
        vv[threadIdx.x] *= temp;

        /* Compute the derivatives for time-average smearing. */
        float sin_HA, cos_HA, sin_Dec, cos_Dec;
        sincosf(gha0_rad, &sin_HA, &cos_HA);
        sincosf(dec0_rad, &sin_Dec, &cos_Dec);
        float xx = (d_x[q] - d_x[p]) * 0.5f;
        float yy = (d_y[q] - d_y[p]) * 0.5f;
        float rot_angle = OMEGA_EARTHf * time_average_sec;
        temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
        du_dt[threadIdx.x] = (xx * cos_HA - yy * sin_HA) * rot_angle;
        dv_dt[threadIdx.x] = temp * sin_Dec;
        dw_dt[threadIdx.x] = -temp * cos_Dec;

        for (int warp = 0; warp < num_warps; ++warp)
        {
            int idx = (threadIdx.x * num_warps) + warp;
            smem_f4c[idx].a = make_float2(0.0f, 0.0f);
            smem_f4c[idx].b = make_float2(0.0f, 0.0f);
            smem_f4c[idx].c = make_float2(0.0f, 0.0f);
            smem_f4c[idx].d = make_float2(0.0f, 0.0f);
        }
    }

    __syncthreads();


    /* Partial sum per thread per baseline. */
    float4c b_sum[BLOCK_SIZE];

    for (int j = 0; j < BLOCK_SIZE; ++j)
    {
        int q = q0 + j;
        if (q > num_stations) continue;
        b_sum[j].a = make_float2(0.0f, 0.0f);
        b_sum[j].b = make_float2(0.0f, 0.0f);
        b_sum[j].c = make_float2(0.0f, 0.0f);
        b_sum[j].d = make_float2(0.0f, 0.0f);
        MArray station_q = &d_Jones[num_sources * q];

        for (int i = threadIdx.x; i < num_sources; i += blockDim.x)
        {
            /* Source l,m,n - common to all baselines */
            const float l = d_l[i];
            const float m = d_m[i];
            const float n = d_n[i];

            const float a = uu[j] * l + vv[j] * m;
            const float rt = oskar_sinc_f(du_dt[j]*l + dv_dt[j]*m + dw_dt[j]*n);
            float rb = oskar_sinc_f(a);
            rb *= rt;

            /* Accumulate baseline visibility response for source. */
            oskar_accumulate_baseline_visibility_for_source_inline_f(&b_sum[j], i,
                    d_I, d_Q, d_U, d_V, station_q, station_p, rb);
        } /* end of loop over source blocks */
    } /* end of loop over baselines */

    for (int j = 0; j < BLOCK_SIZE; ++j)
    {
        // Warp shuffle reduction of source block for baseline.
        for (int i = warpSize/2; i >=1; i/=2) {
            b_sum[j].a.x += __shfl_xor(b_sum[j].a.x, i, warpSize);
            b_sum[j].a.y += __shfl_xor(b_sum[j].a.y, i, warpSize);
            b_sum[j].b.x += __shfl_xor(b_sum[j].b.x, i, warpSize);
            b_sum[j].b.y += __shfl_xor(b_sum[j].b.y, i, warpSize);
            b_sum[j].c.x += __shfl_xor(b_sum[j].c.x, i, warpSize);
            b_sum[j].c.y += __shfl_xor(b_sum[j].c.y, i, warpSize);
            b_sum[j].d.x += __shfl_xor(b_sum[j].d.x, i, warpSize);
            b_sum[j].d.y += __shfl_xor(b_sum[j].d.y, i, warpSize);
        }
        // Store the partial sum (per warp) into shared memory.
        int laneID = threadIdx.x & 0x1f;
        if (laneID == 0) {
            int warp = threadIdx.x / warpSize;
            int idx = (j * num_warps) + warp;
            smem_f4c[idx].a.x += b_sum[j].a.x;
            smem_f4c[idx].a.y += b_sum[j].a.y;
            smem_f4c[idx].b.x += b_sum[j].b.x;
            smem_f4c[idx].b.y += b_sum[j].b.y;
            smem_f4c[idx].c.x += b_sum[j].c.x;
            smem_f4c[idx].c.y += b_sum[j].c.y;
            smem_f4c[idx].d.x += b_sum[j].d.x;
            smem_f4c[idx].d.y += b_sum[j].d.y;
        }
    }

    // Make sure all threads have finished.
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x < BLOCK_SIZE)
    {
        int q = q0 + threadIdx.x;
        if (q < num_stations) {

            /* Sum over all sources for this baseline. */
            b_sum[threadIdx.x].a = make_float2(0.0, 0.0);
            b_sum[threadIdx.x].b = make_float2(0.0, 0.0);
            b_sum[threadIdx.x].c = make_float2(0.0, 0.0);
            b_sum[threadIdx.x].d = make_float2(0.0, 0.0);

            int num_warps = blockDim.x/warpSize;
            for (int warp = 0; warp < num_warps; ++warp)
            {
                int idx = (threadIdx.x * num_warps) + warp;
                b_sum[threadIdx.x].a.x += smem_f4c[idx].a.x;
                b_sum[threadIdx.x].a.y += smem_f4c[idx].a.y;
                b_sum[threadIdx.x].b.x += smem_f4c[idx].b.x;
                b_sum[threadIdx.x].b.y += smem_f4c[idx].b.y;
                b_sum[threadIdx.x].c.x += smem_f4c[idx].c.x;
                b_sum[threadIdx.x].c.y += smem_f4c[idx].c.y;
                b_sum[threadIdx.x].d.x += smem_f4c[idx].d.x;
                b_sum[threadIdx.x].d.y += smem_f4c[idx].d.y;
            }

            /* Determine 1D visibility index for global memory store. */
            int ipq = p*(num_stations-1) - (p-1)*p/2 + q - p - 1;
            /* Add result of this thread block to the baseline visibility. */
            d_vis[ipq].a.x += b_sum[threadIdx.x].a.x;
            d_vis[ipq].a.y += b_sum[threadIdx.x].a.y;
            d_vis[ipq].b.x += b_sum[threadIdx.x].b.x;
            d_vis[ipq].b.y += b_sum[threadIdx.x].b.y;
            d_vis[ipq].c.x += b_sum[threadIdx.x].c.x;
            d_vis[ipq].c.y += b_sum[threadIdx.x].c.y;
            d_vis[ipq].d.x += b_sum[threadIdx.x].d.x;
            d_vis[ipq].d.y += b_sum[threadIdx.x].d.y;
        }
    }
#endif /* __CUDA_ARCH__ >= 300 */
}

void correlate_warpshuffle_blocked_f(int num_sources, int num_stations, const float4c* d_Jones,
        const float* d_I, const float* d_Q, const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n, const float* d_u,
        const float* d_v, const float* d_x, const float* d_y,
        float freq, float bandwidth, float time_int, float gha0, float dec0,
        float4c* d_vis)
{
    int num_blocks = ((num_stations-1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(num_blocks, num_stations-1);
    dim3 blockDim(NUM_THREADS, 1);
    int num_warps = blockDim.x / 32;
    size_t shared_mem = (num_warps * BLOCK_SIZE) * sizeof(float4c);
    cudaFuncSetCacheConfig(correlate_cudak_f, cudaFuncCachePreferShared);
    correlate_warpshuffle_blocked_cudak_f
    OSKAR_CUDAK_CONF(gridDim, blockDim, shared_mem)
    (num_sources, num_stations, d_Jones,
            d_I, d_Q, d_U, d_V, d_l, d_m, d_n, d_u, d_v, d_x, d_y,
            freq, bandwidth, time_int, gha0, dec0, d_vis);
}

// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__global__ void
//__launch_bounds__(128, 16)
correlate_warpshuffle_blocked_cudak_f(const int num_sources, const int num_stations,
        MArray d_Jones, Array d_I, Array d_Q, Array d_U,
        Array d_V, Array d_l, Array d_m, Array d_n, Array d_u, Array d_v,
        Array d_x, Array d_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* d_vis)
{
#if __CUDA_ARCH__ >= 300
    // This kernel evaluates baselines p-q for a single station p and a range
    // of stations q, determined by the BLOCK_SIZE

    // Index of first station q in the block.
    int q0 = (blockIdx.y+1) + (blockIdx.x*BLOCK_SIZE);

    // Return immediately if the block is outside range of valid baselines.
    if (q0 > num_stations)
        return;

    // Station index p (common to all baselines processed by this block)
    int p = blockIdx.y;

    // TODO fix this for the case where blockDim doesn't divide perfectly by
    // warp size
    int num_warps = blockDim.x/warpSize;

    /* Cache common (per baseline values). */
    float uu[BLOCK_SIZE], vv[BLOCK_SIZE];
    float du_dt[BLOCK_SIZE], dv_dt[BLOCK_SIZE], dw_dt[BLOCK_SIZE];

    /* Use the first BLOCK_SIZE threads to cache the per baseline values:
     * uu, vv, du_dt, dv_dt, & dw_dt */
    if (threadIdx.x < BLOCK_SIZE)
    {
        int q = q0 + threadIdx.x; // index of station q in the baseline

        /* Baseline lengths. */
        uu[threadIdx.x] = (d_u[q] - d_u[p]) * 0.5f;
        vv[threadIdx.x] = (d_v[q] - d_v[p]) * 0.5f;

        /* Modify the baseline distance to include the common components
         * of the bandwidth smearing term. */
        float temp = channel_bandwidth_hz / freq_hz; /* Fractional bandwidth */
        uu[threadIdx.x] *= temp;
        vv[threadIdx.x] *= temp;

        /* Compute the derivatives for time-average smearing. */
        float sin_HA, cos_HA, sin_Dec, cos_Dec;
        sincosf(gha0_rad, &sin_HA, &cos_HA);
        sincosf(dec0_rad, &sin_Dec, &cos_Dec);
        float xx = (d_x[q] - d_x[p]) * 0.5f;
        float yy = (d_y[q] - d_y[p]) * 0.5f;
        float rot_angle = OMEGA_EARTHf * time_average_sec;
        temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
        du_dt[threadIdx.x] = (xx * cos_HA - yy * sin_HA) * rot_angle;
        dv_dt[threadIdx.x] = temp * sin_Dec;
        dw_dt[threadIdx.x] = -temp * cos_Dec;

        for (int warp = 0; warp < num_warps; ++warp)
        {
            int idx = (threadIdx.x * num_warps) + warp;
            smem_f4c[idx].a = make_float2(0.0f, 0.0f);
            smem_f4c[idx].b = make_float2(0.0f, 0.0f);
            smem_f4c[idx].c = make_float2(0.0f, 0.0f);
            smem_f4c[idx].d = make_float2(0.0f, 0.0f);
        }
    }

    __syncthreads();


    /* Partial sum per thread per baseline. */
    float4c b_sum[BLOCK_SIZE];

    MArray station_p = &d_Jones[num_sources * p];

    for (int j = 0; j < BLOCK_SIZE; ++j)
    {
        int q = q0 + j;
        if (q > num_stations) continue;
        b_sum[j].a = make_float2(0.0f, 0.0f);
        b_sum[j].b = make_float2(0.0f, 0.0f);
        b_sum[j].c = make_float2(0.0f, 0.0f);
        b_sum[j].d = make_float2(0.0f, 0.0f);
        MArray station_q = &d_Jones[num_sources * q];

        for (int i = threadIdx.x; i < num_sources; i += blockDim.x)
        {
            /* Source l,m,n - common to all baselines */
            float l = d_l[i];
            float m = d_m[i];
            float n = d_n[i];

            float a = uu[j] * l + vv[j] * m;
            float rb = oskar_sinc_f(a);
            float rt = oskar_sinc_f(du_dt[j]*l + dv_dt[j]*m + dw_dt[j]*n);
            rb *= rt;

            /* Accumulate baseline visibility response for source. */
            oskar_accumulate_baseline_visibility_for_source_inline_f(&b_sum[j], i,
                    d_I, d_Q, d_U, d_V, station_q, station_p, rb);
        } /* end of loop over source blocks */
    } /* end of loop over baselines */

    for (int j = 0; j < BLOCK_SIZE; ++j)
    {
        // Warp shuffle reduction of source block for baseline.
        for (int i = warpSize/2; i >=1; i/=2) {
            b_sum[j].a.x += __shfl_xor(b_sum[j].a.x, i, warpSize);
            b_sum[j].a.y += __shfl_xor(b_sum[j].a.y, i, warpSize);
            b_sum[j].b.x += __shfl_xor(b_sum[j].b.x, i, warpSize);
            b_sum[j].b.y += __shfl_xor(b_sum[j].b.y, i, warpSize);
            b_sum[j].c.x += __shfl_xor(b_sum[j].c.x, i, warpSize);
            b_sum[j].c.y += __shfl_xor(b_sum[j].c.y, i, warpSize);
            b_sum[j].d.x += __shfl_xor(b_sum[j].d.x, i, warpSize);
            b_sum[j].d.y += __shfl_xor(b_sum[j].d.y, i, warpSize);
        }
        // Store the partial sum (per warp) into shared memory.
        int laneID = threadIdx.x & 0x1f;
        if (laneID == 0) {
            int warp = threadIdx.x / warpSize;
            int idx = (j * num_warps) + warp;
            smem_f4c[idx].a.x += b_sum[j].a.x;
            smem_f4c[idx].a.y += b_sum[j].a.y;
            smem_f4c[idx].b.x += b_sum[j].b.x;
            smem_f4c[idx].b.y += b_sum[j].b.y;
            smem_f4c[idx].c.x += b_sum[j].c.x;
            smem_f4c[idx].c.y += b_sum[j].c.y;
            smem_f4c[idx].d.x += b_sum[j].d.x;
            smem_f4c[idx].d.y += b_sum[j].d.y;
        }
    }

    // Make sure all threads have finished.
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x < BLOCK_SIZE)
    {
        int q = q0 + threadIdx.x;
        if (q < num_stations) {

            /* Sum over all sources for this baseline. */
            b_sum[threadIdx.x].a = make_float2(0.0, 0.0);
            b_sum[threadIdx.x].b = make_float2(0.0, 0.0);
            b_sum[threadIdx.x].c = make_float2(0.0, 0.0);
            b_sum[threadIdx.x].d = make_float2(0.0, 0.0);

            int num_warps = blockDim.x/warpSize;
            for (int warp = 0; warp < num_warps; ++warp)
            {
                int idx = (threadIdx.x * num_warps) + warp;
                b_sum[threadIdx.x].a.x += smem_f4c[idx].a.x;
                b_sum[threadIdx.x].a.y += smem_f4c[idx].a.y;
                b_sum[threadIdx.x].b.x += smem_f4c[idx].b.x;
                b_sum[threadIdx.x].b.y += smem_f4c[idx].b.y;
                b_sum[threadIdx.x].c.x += smem_f4c[idx].c.x;
                b_sum[threadIdx.x].c.y += smem_f4c[idx].c.y;
                b_sum[threadIdx.x].d.x += smem_f4c[idx].d.x;
                b_sum[threadIdx.x].d.y += smem_f4c[idx].d.y;
            }

            /* Determine 1D visibility index for global memory store. */
            int ipq = p*(num_stations-1) - (p-1)*p/2 + q - p - 1;
            /* Add result of this thread block to the baseline visibility. */
            d_vis[ipq].a.x += b_sum[threadIdx.x].a.x;
            d_vis[ipq].a.y += b_sum[threadIdx.x].a.y;
            d_vis[ipq].b.x += b_sum[threadIdx.x].b.x;
            d_vis[ipq].b.y += b_sum[threadIdx.x].b.y;
            d_vis[ipq].c.x += b_sum[threadIdx.x].c.x;
            d_vis[ipq].c.y += b_sum[threadIdx.x].c.y;
            d_vis[ipq].d.x += b_sum[threadIdx.x].d.x;
            d_vis[ipq].d.y += b_sum[threadIdx.x].d.y;
        }
    }
#endif /* __CUDA_ARCH__ >= 300 */
}









//__global__
//void test_warp_shuffle(float* d_warp_sum)
//{
//    float value = (float)threadIdx.x;
//    for (int i = 16; i >=1; i/=2) {
//        value += __shfl_xor(value, i, warpSize);
//    }
//    d_warp_sum[0] = value;
//}
//


void print_float4c(float4c value)
{
    printf("(%.2f %.2fi) (%.2f %.2fi)\n"
            "(%.2f %.2fi) (%.2f %.2fi)\n",
            value.a.x, value.a.y,
            value.b.x, value.b.y,
            value.c.x, value.c.y,
            value.d.x, value.d.y);
}

void print_float4c_range(int i0, int i1, float4c* values)
{
    for (int i = i0; i < i1; ++i) {
        printf("[%04i]  (%.2f %.2fi) (%.2f %.2fi)\n"
                "        (%.2f %.2fi) (%.2f %.2fi)\n",
                i,
                values[i].a.x, values[i].a.y,
                values[i].b.x, values[i].b.y,
                values[i].c.x, values[i].c.y,
                values[i].d.x, values[i].d.y);
    }
}


