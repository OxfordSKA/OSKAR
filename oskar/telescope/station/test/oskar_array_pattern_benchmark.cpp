/*
 * Copyright (c) 2013-2014, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "apps/oskar_OptionParser.h"
#include "math/oskar_cmath.h"
#include "mem/oskar_mem.h"
#include "telescope/station/oskar_station.h"
#include "telescope/station/private_station.h"
#include "telescope/station/oskar_evaluate_array_pattern.h"
#include "telescope/station/oskar_evaluate_array_pattern_hierarchical.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"
#include "oskar_version.h"

#include <cstdlib>
#include <cstdio>

enum OpType { O2C, C2C, M2M, UNDEF };

int benchmark(int num_elements, int num_directions, OpType op_type,
        int loc, int precision, bool evaluate_2d, int niter, double& time_taken);


int main(int argc, char** argv)
{
    oskar::OptionParser opt("oskar_array_pattern_benchmark", OSKAR_VERSION_STR);
    opt.add_required("No. array elements");
    opt.add_required("No. directions");
    opt.add_flag("-sp", "Use single precision (default: double precision)");
    opt.add_flag("-g", "Run on the GPU");
    opt.add_flag("-c", "Run on the CPU");
    opt.add_flag("-o2c", "Single level beam pattern, phase only (real to complex DFT)");
    opt.add_flag("-c2c", "Beam pattern using complex inputs (complex to complex DFT)");
    opt.add_flag("-m2m", "Beam pattern using complex, polarised inputs (complex matrix to matrix DFT)");
    opt.add_flag("-2d", "Use a 2-dimensional phase term (default: 3D)");
    opt.add_flag("-n", "Number of iterations", 1, "1");
    opt.add_flag("-v", "Display verbose output.");

    if (!opt.check_options(argc, argv))
        return EXIT_FAILURE;

    int num_elements = atoi(opt.get_arg(0));
    int num_directions = atoi(opt.get_arg(1));
    OpType op_type = UNDEF;
    int op_type_count = 0;
    if (opt.is_set("-o2c"))
    {
        op_type = O2C;
        op_type_count++;
    }
    if (opt.is_set("-c2c"))
    {
        op_type = C2C;
        op_type_count++;
    }
    if (opt.is_set("-m2m"))
    {
        op_type = M2M;
        op_type_count++;
    }

    int loc;
    if (opt.is_set("-g"))
        loc = OSKAR_GPU;
    if (opt.is_set("-c"))
        loc = OSKAR_CPU;
    if (!(opt.is_set("-c") ^ opt.is_set("-g")))
    {
        opt.error("Please select one of -g or -c");
        return EXIT_FAILURE;
    }

    int precision = opt.is_set("-sp") ? OSKAR_SINGLE : OSKAR_DOUBLE;
    bool evaluate_2d = opt.is_set("-2d") ? true : false;
    int niter;
    opt.get("-n")->getInt(niter);

    if (op_type == UNDEF || op_type_count != 1) {
        opt.error("Please select one of the following flags: -o2c, -c2c, -m2m");
        return EXIT_FAILURE;
    }

    if (opt.is_set("-v"))
    {
        printf("\n");
        printf("- Number of elements: %i\n", num_elements);
        printf("- Number of directions: %i\n", num_directions);
        printf("- Precision: %s\n", (precision == OSKAR_SINGLE) ? "single" : "double");
        printf("- %s\n", loc == OSKAR_CPU ? "CPU" : "GPU");
        printf("- %s\n", evaluate_2d ? "2D" : "3D");
        printf("- Evaluation type = ");
        if (op_type == O2C) printf("o2c\n");
        else if (op_type == C2C) printf("c2c\n");
        else if (op_type == M2M) printf("m2m\n");
        else printf("Error undefined!\n");
        printf("- Number of iterations: %i\n", niter);
        printf("\n");
    }

    double time_taken = 0.0;
    int status = benchmark(num_elements, num_directions, op_type, loc,
            precision, evaluate_2d, niter, time_taken);

    if (status) {
        fprintf(stderr, "ERROR: array pattern evaluation failed with code %i: "
                "%s\n", status, oskar_get_error_string(status));
        return EXIT_FAILURE;
    }
    if (opt.is_set("-v"))
    {
        printf("==> Total time taken: %f seconds.\n", time_taken);
        printf("==> Time taken per iteration: %f seconds.\n", time_taken/niter);
        printf("\n");
    }
    else {
        printf("%f\n", time_taken/niter);
    }

    return EXIT_SUCCESS;
}


int benchmark(int num_elements, int num_directions, OpType op_type,
        int loc, int precision, bool evaluate_2d, int niter, double& time_taken)
{
    int status = 0;

    // Create the timer.
    oskar_Timer *tmr = oskar_timer_create(OSKAR_TIMER_CUDA);

    oskar_Station* station = oskar_station_create(precision, loc,
            num_elements, &status);
    if (status) return status;
    station->array_is_3d = (evaluate_2d) ? OSKAR_FALSE : OSKAR_TRUE;

    oskar_Mem *x, *y, *z, *weights = 0, *beam = 0, *signal = 0;
    x = oskar_mem_create(precision, loc, num_directions, &status);
    y = oskar_mem_create(precision, loc, num_directions, &status);
    z = oskar_mem_create(precision, loc, num_directions, &status);
    if (status) return status;

    if (op_type == O2C)
    {
        int type = precision | OSKAR_COMPLEX;
        beam = oskar_mem_create(type, loc, num_directions, &status);
        weights = oskar_mem_create(type, loc, num_elements, &status);
        if (status) return status;

        oskar_timer_start(tmr);
        for (int i = 0; i < niter; ++i)
        {
            oskar_evaluate_array_pattern(beam, 2.0 * M_PI, station,
                    num_directions, x, y, z, weights, &status);
        }
        time_taken = oskar_timer_elapsed(tmr);
    }
    else if (op_type == C2C || op_type == M2M)
    {
        int type = precision | OSKAR_COMPLEX;
        int num_signals = num_directions * num_elements;

        weights = oskar_mem_create(type, loc, num_elements, &status);
        if (op_type == C2C)
        {
            beam = oskar_mem_create(type, loc, num_directions, &status);
            signal = oskar_mem_create(type, loc, num_signals, &status);
        }
        else
        {
            type |= OSKAR_MATRIX;
            beam = oskar_mem_create(type, loc, num_directions, &status);
            signal = oskar_mem_create(type, loc, num_signals, &status);
        }
        if (status) return status;

        oskar_timer_start(tmr);
        for (int i = 0; i < niter; ++i)
        {
            oskar_evaluate_array_pattern_hierarchical(beam, 2.0 * M_PI, station,
                    num_directions, x, y, z, signal, weights, &status);
        }
        time_taken = oskar_timer_elapsed(tmr);
    }

    // Destroy the timer.
    oskar_timer_free(tmr);

    // Free memory.
    oskar_station_free(station, &status);
    oskar_mem_free(x, &status);
    oskar_mem_free(y, &status);
    oskar_mem_free(z, &status);
    oskar_mem_free(weights, &status);
    oskar_mem_free(beam, &status);
    oskar_mem_free(signal, &status);

    return status;
}
