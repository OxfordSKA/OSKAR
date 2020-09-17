/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct ThreadArgs
{
    oskar_Interferometer* h;
    DeviceData* d;
    int num_threads, thread_id, *status;
};
typedef struct ThreadArgs ThreadArgs;

static void* run_blocks(void* arg)
{
    oskar_Interferometer* h;
    int b, *status;

    /* Get thread function arguments. */
    h = ((ThreadArgs*)arg)->h;
    const int num_threads = ((ThreadArgs*)arg)->num_threads;
    const int thread_id = ((ThreadArgs*)arg)->thread_id;
    const int device_id = thread_id - 1;
    status = ((ThreadArgs*)arg)->status;

#ifdef _OPENMP
    /* Disable any nested parallelism. */
    omp_set_nested(0);
    omp_set_num_threads(1);
#endif

    /* Loop over visibility blocks, running simulation and file
     * writing one block at a time. Simulation and file output are overlapped
     * by using double buffering, and a dedicated thread is used for file
     * output.
     *
     * Thread 0 is used for file writes.
     * Threads 1 to n (mapped to compute devices) do the simulation.
     *
     * Note that no write is launched on the first loop counter (as no
     * data are ready yet) and no simulation is performed for the last loop
     * counter (which corresponds to the last block + 1) as this iteration
     * simply writes the last block.
     */
    const int num_blocks = oskar_interferometer_num_vis_blocks(h);
    for (b = 0; b < num_blocks + 1; ++b)
    {
        if ((thread_id > 0 || num_threads == 1) && b < num_blocks)
            oskar_interferometer_run_block(h, b, device_id, status);
        if (thread_id == 0 && b > 0)
        {
            oskar_VisBlock* block;
            block = oskar_interferometer_finalise_block(h, b - 1, status);
            oskar_interferometer_write_block(h, block, b - 1, status);
        }

        /* Barrier 1: Reset work unit index and print status. */
        oskar_barrier_wait(h->barrier);
        if (thread_id == 0)
            oskar_interferometer_reset_work_unit_index(h);

        /* Barrier 2: Synchronise before moving to the next block. */
        oskar_barrier_wait(h->barrier);
    }
    return 0;
}


void oskar_interferometer_run(oskar_Interferometer* h, int* status)
{
    int i;
    oskar_Thread** threads = 0;
    ThreadArgs* args = 0;
    if (*status || !h) return;

    /* Check the visibilities are going somewhere. */
    if (!h->vis_name
#ifndef OSKAR_NO_MS
            && !h->ms_name
#endif
    )
    {
        oskar_log_error(h->log, "No output file specified.");
#ifdef OSKAR_NO_MS
        if (h->ms_name)
            oskar_log_error(h->log,
                    "OSKAR was compiled without Measurement Set support.");
#endif
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Initialise if required. */
    oskar_interferometer_check_init(h, status);

    /* Set up worker threads. */
    const int num_threads = h->num_devices + 1;
    oskar_barrier_set_num_threads(h->barrier, num_threads);
    threads = (oskar_Thread**) calloc(num_threads, sizeof(oskar_Thread*));
    args = (ThreadArgs*) calloc(num_threads, sizeof(ThreadArgs));
    for (i = 0; i < num_threads; ++i)
    {
        args[i].h = h;
        args[i].num_threads = num_threads;
        args[i].thread_id = i;
        args[i].status = status;
    }

    /* Start the worker threads. */
    oskar_interferometer_reset_work_unit_index(h);
    for (i = 0; i < num_threads; ++i)
        threads[i] = oskar_thread_create(run_blocks, (void*)&args[i], 0);

    /* Wait for worker threads to finish. */
    for (i = 0; i < num_threads; ++i)
    {
        oskar_thread_join(threads[i]);
        oskar_thread_free(threads[i]);
    }
    free(threads);
    free(args);

    /* Finalise. */
    oskar_interferometer_finalise(h, status);
}

#ifdef __cplusplus
}
#endif
