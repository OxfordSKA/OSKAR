/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_option_parser.h"
#include "math/oskar_cmath.h"
#include "math/oskar_dftw.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_device.h"
#include "oskar_version.h"

#include <cstdlib>
#include <cstdio>

enum OpType { C2C, M2M, UNDEF };

int benchmark(int num_elements, int num_directions, int num_element_types,
        OpType op_type, int loc, int precision, bool evaluate_2d, int niter,
        double& time_taken);


int main(int argc, char** argv)
{
    oskar::OptionParser opt("oskar_array_pattern_benchmark", OSKAR_VERSION_STR);
    opt.add_required("No. array elements");
    opt.add_required("No. directions");
    opt.add_flag("-sp", "Use single precision (default: double precision)");
    opt.add_flag("-g", "Run on the GPU");
    opt.add_flag("-c", "Run on the CPU");
    opt.add_flag("-cl", "Run using OpenCL");
    opt.add_flag("-c2c", "Beam pattern using complex inputs (complex to complex DFT)");
    opt.add_flag("-m2m", "Beam pattern using complex, polarised inputs (complex matrix to matrix DFT)");
    opt.add_flag("-2d", "Use a 2-dimensional phase term (default: 3D)");
    opt.add_flag("-n", "Number of iterations", 1, "1");
    opt.add_flag("-e", "Number of element types", 1, "1");
    opt.add_flag("-v", "Display verbose output.");

    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;

    int num_elements = atoi(opt.get_arg(0));
    int num_directions = atoi(opt.get_arg(1));
    OpType op_type = UNDEF;
    int op_type_count = 0;
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

    int location = -1;
    if (opt.is_set("-g")) location = OSKAR_GPU;
    if (opt.is_set("-c")) location = OSKAR_CPU;
    if (opt.is_set("-cl")) location = OSKAR_CL;
    if (location < 0)
    {
        opt.error("Please select one of -g, -c or -cl");
        return EXIT_FAILURE;
    }

    int precision = opt.is_set("-sp") ? OSKAR_SINGLE : OSKAR_DOUBLE;
    bool evaluate_2d = opt.is_set("-2d") ? true : false;
    int niter = opt.get_int("-n");
    int num_element_types = opt.get_int("-e");

    if (op_type == UNDEF || op_type_count != 1)
    {
        opt.error("Please select one of the following flags: -o2c, -c2c, -m2m");
        return EXIT_FAILURE;
    }

    if (opt.is_set("-v"))
    {
        printf("\n");
        printf("- Number of elements: %i\n", num_elements);
        printf("- Number of directions: %i\n", num_directions);
        printf("- Precision: %s\n", (precision == OSKAR_SINGLE) ? "single" : "double");
        printf("- %s\n", evaluate_2d ? "2D" : "3D");
        printf("- Operation type: ");
        if (op_type == C2C)
        {
            printf("c2c\n");
        }
        else if (op_type == M2M)
        {
            printf("m2m\n");
        }
        else
        {
            printf("Error undefined!\n");
        }
        printf("- Number of iterations: %i\n", niter);
        printf("\n");
    }

    double time_taken = 0.0;
    oskar_device_set_require_double_precision(precision == OSKAR_DOUBLE);
    int status = benchmark(num_elements, num_directions, num_element_types,
            op_type, location, precision, evaluate_2d, niter, time_taken);

    if (status)
    {
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
    else
    {
        printf("%f\n", time_taken/niter);
    }

    return EXIT_SUCCESS;
}


int benchmark(int num_elements, int num_directions, int num_element_types,
        OpType op_type, int loc, int precision, bool evaluate_2d, int niter,
        double& time_taken)
{
    int status = 0;
    int type = precision | OSKAR_COMPLEX;
    oskar_Mem *beam = 0, *signal = 0, *z = 0, *z_i = 0;
    oskar_Mem *x = oskar_mem_create(precision, loc, num_directions, &status);
    oskar_Mem *y = oskar_mem_create(precision, loc, num_directions, &status);
    oskar_Mem *x_i = oskar_mem_create(precision, loc, num_elements, &status);
    oskar_Mem *y_i = oskar_mem_create(precision, loc, num_elements, &status);
    oskar_Mem *weights = oskar_mem_create(type, loc, num_elements, &status);
    oskar_Mem *element_types_cpu = oskar_mem_create(OSKAR_INT, OSKAR_CPU,
            num_elements, &status);
    int* el_type = oskar_mem_int(element_types_cpu, &status);
    for (int j = 0; j < num_elements; j += num_element_types)
    {
        for (int i = 0; i < num_element_types; ++i)
        {
            if (i + j >= num_elements) break;
            el_type[i + j] = i;
        }
    }
    if (!evaluate_2d)
    {
        z = oskar_mem_create(precision, loc, num_directions, &status);
        z_i = oskar_mem_create(precision, loc, num_elements, &status);
    }
    int num_signals = num_directions * num_elements;
    if (op_type == M2M) type |= OSKAR_MATRIX;
    beam = oskar_mem_create(type, loc, num_directions, &status);
    signal = oskar_mem_create(type, loc, num_signals, &status);
    oskar_Mem* element_types = oskar_mem_create_copy(
            element_types_cpu, loc, &status);

    oskar_Timer *tmr = oskar_timer_create(loc);
    if (!status)
    {
        char* device_name = oskar_device_name(loc, 0);
        printf("Using device '%s'\n", device_name);
        free(device_name);
        oskar_timer_start(tmr);
        for (int i = 0; i < niter; ++i)
        {
            oskar_dftw(0, num_elements, 2.0 * M_PI, weights, x_i, y_i, z_i,
                    0, num_directions, x, y, z,
                    element_types, signal, 1, 1, 0, beam, &status);
        }
        time_taken = oskar_timer_elapsed(tmr);
    }

    // Free memory.
    oskar_timer_free(tmr);
    oskar_mem_free(x, &status);
    oskar_mem_free(y, &status);
    oskar_mem_free(z, &status);
    oskar_mem_free(x_i, &status);
    oskar_mem_free(y_i, &status);
    oskar_mem_free(z_i, &status);
    oskar_mem_free(weights, &status);
    oskar_mem_free(element_types, &status);
    oskar_mem_free(element_types_cpu, &status);
    oskar_mem_free(beam, &status);
    oskar_mem_free(signal, &status);

    return status;
}
