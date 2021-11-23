/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "utility/oskar_get_num_procs.h"
#include "utility/oskar_thread.h"
#include "utility/oskar_timer.h"
#include <cstdlib>

#define ENABLE_PRINT 1

void print_from_thread(int loop, int id, const char* message)
{
    if (ENABLE_PRINT)
    {
        printf("(Loop %d, Thread %2d): %s\n", loop, id, message);
    }
}

void* thread_simple(void* arg)
{
    size_t thread_id = (size_t)arg;
    print_from_thread(0, (int)thread_id, "Hello");
    return 0;
}

struct ThreadArgs
{
    int thread_id;
    oskar_Barrier* barrier;
};
typedef struct ThreadArgs ThreadArgs;

void* thread_barriers(void* arg)
{
    ThreadArgs* args = (ThreadArgs*) arg;
    int thread_id = args->thread_id;
    for (int i = 0; i < 4; ++i)
    {
        print_from_thread(i, thread_id, "Hello");
        oskar_barrier_wait(args->barrier);
        print_from_thread(i, thread_id, "Goodbye");
        oskar_barrier_wait(args->barrier);
    }
    return 0;
}

TEST(thread, create_and_join)
{
    // Get the number of CPU cores.
    size_t num_threads = (size_t) oskar_get_num_procs();

    // Allocate thread array.
    oskar_Thread** threads = (oskar_Thread**)
            calloc(num_threads, sizeof(oskar_Thread*));

    // Start all the threads.
    for (size_t i = 0; i < num_threads; ++i)
    {
        threads[i] = oskar_thread_create(thread_simple, (void*)i, 0);
    }

    // Wait for all threads to finish.
    for (size_t i = 0; i < num_threads; ++i)
    {
        oskar_thread_join(threads[i]);
    }

    // Clean up.
    for (size_t i = 0; i < num_threads; ++i)
    {
        oskar_thread_free(threads[i]);
    }
    free(threads);
}

TEST(thread, barriers)
{
    // Set the number of threads.
    int num_threads = 8;

    // Create the shared barrier.
    oskar_Barrier* barrier = oskar_barrier_create(num_threads);

    // Allocate thread array and thread arguments for each thread.
    oskar_Thread** threads = (oskar_Thread**)
            calloc((size_t) num_threads, sizeof(oskar_Thread*));
    ThreadArgs* args = (ThreadArgs*)
            calloc((size_t) num_threads, sizeof(ThreadArgs));

    // Start all the threads.
    for (int i = 0; i < num_threads; ++i)
    {
        args[i].barrier = barrier;
        args[i].thread_id = (int) i;
        threads[i] = oskar_thread_create(thread_barriers, (void*)(&args[i]), 0);
    }

    // Wait for all threads to finish.
    for (int i = 0; i < num_threads; ++i)
    {
        oskar_thread_join(threads[i]);
    }

    // Clean up.
    for (int i = 0; i < num_threads; ++i)
    {
        oskar_thread_free(threads[i]);
    }
    oskar_barrier_free(barrier);
    free(args);
    free(threads);
}
