#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <sys/time.h>

#define TIMER_START {struct timeval _t1; gettimeofday(&_t1, NULL);
#define TIMER_STOP(...) struct timeval _t2; gettimeofday(&_t2, NULL); \
    double _start = _t1.tv_sec + _t1.tv_usec * 1.0e-6; \
    double _end = _t2.tv_sec + _t2.tv_usec * 1.0e-6; \
    fprintf(stdout, "\n"); \
    fprintf(stdout, __VA_ARGS__); \
    fprintf(stdout, ": %.6f sec.\n", _end - _start);};

extern "C" {
void bispev_(float tx[], int* nx, float ty[], int* ny, float c[],
        int* kx, int* ky, float x[], int* mx, float y[], int* my,
        float z[], float wrk[], int* lwrk, int iwrk[], int* kwrk, int* ier);

void regrid_(int* iopt, int* mx, float x[], int* my, float y[], float z[],
        float* xb, float* xe, float* yb, float* ye, int* kx, int* ky,
        float* s, int* nxest, int* nyest, int* nx, float tx[], int* ny,
        float ty[], float c[], float* fp, float wrk[], int* lwrk, int iwrk[],
        int* kwrk, int* ier);
}

using std::vector;

int main(int argc, char** argv)
{
    // Get input filename from command line arguments.
    if (argc != 2)
    {
        fprintf(stderr, "Specify filename of look-up table.\n");
        return 1;
    }
    const char* filename = argv[1];

    // Open input file.
    FILE* file = fopen(filename, "r");

    // Read header.
    vector<int> header(8);
    size_t t = fread(&header[0], sizeof(int), header.size(), file);
    if (t != header.size())
    {
        fprintf(stderr, "Error reading file header.\n");
        return 1;
    }

    // Get data dimensions.
    int size_x = header[0];
    int size_y = header[1];
    if (header[2] != sizeof(float))
    {
        fprintf(stderr, "Bad data type, or corrupted header.\n");
        return 1;
    }

    // Read the data.
    vector<float> data(size_x * size_y);
    t = fread(&data[0], sizeof(float), data.size(), file);
    if (t != data.size())
    {
        fprintf(stderr, "Error reading file data.\n");
        return 1;
    }

    // Close input file.
    fclose(file);

    // Create the data axes.
    vector<float> x(size_x), y(size_y);
    for (int i = 0; i < size_x; ++i)
        x[i] = (float) i;
    for (int i = 0; i < size_y; ++i)
        y[i] = (float) i;

    // Set up the surface fitting parameters.
    int iopt = 0; // -1 = Specify least-squares spline.
    int kxy = 3; // Degree of spline (cubic).
    float noise = 5e-4; // Numerical noise on input data.

    // Checks.
    if (size_x <= kxy || size_y <= kxy)
    {
        fprintf(stderr, "ERROR: Input grid dimensions too small. Aborting.\n");
        return 1;
    }

    // Set up the spline knots.
    int nx = 0; // Number of knots in x.
    int ny = 0; // Number of knots in y.
    int nxest = size_x + kxy + 1; // Maximum number of knots in x.
    int nyest = size_y + kxy + 1; // Maximum number of knots in y.
    vector<float> tx(nxest, 0.0); // Spline knots in x.
    vector<float> ty(nyest, 0.0); // Spline knots in y.
    vector<float> c((nxest-kxy-1)*(nyest-kxy-1)); // Output spline coefficients.
    float fp = 0.0; // Output sum of squared residuals of spline approximation.

    // Set up workspace.
    int u = size_y > nxest ? size_y : nxest;
    int lwrk = 4 + nxest * (size_y + 2 * kxy + 5) +
            nyest * (2 * kxy + 5) + size_x * (kxy + 1) +
            size_y * (kxy + 1) + u;
    vector<float> wrk(lwrk);
    int kwrk = 3 + size_x + size_y + nxest + nyest;
    vector<int> iwrk(kwrk);
    int ier = 0; // Output return code.

    TIMER_START
    int k = 0;
    // Set initial smoothing factor (ignored for iopt < 0).
    float s = size_x * size_y * pow(noise, 2.0);
    bool fail = false;
    do
    {
        // Generate knot positions *at grid points* if required.
        if (iopt < 0 || fail)
        {
            int i, k, stride = 1;
            for (k = 0, i = kxy - 1; i <= size_x - kxy + stride; i += stride, ++k)
                tx[k + kxy + 1] = x[i]; // Knot x positions.
            nx = k + 2 * kxy + 1;
            for (k = 0, i = kxy - 1; i <= size_y - kxy + stride; i += stride, ++k)
                ty[k + kxy + 1] = y[i]; // Knot y positions.
            ny = k + 2 * kxy + 1;
        }

        // Set iopt to 1 if this is at least the second of multiple passes.
        if (k > 0) iopt = 1;
        regrid_(&iopt, &size_x, &x[0], &size_y, &y[0], &data[0],
                &x[0], &x[size_x-1], &y[0], &y[size_y-1], &kxy, &kxy, &s,
                &nxest, &nyest, &nx, &tx[0], &ny, &ty[0], &c[0], &fp,
                &wrk[0], &lwrk, &iwrk[0], &kwrk, &ier);
        if (ier == 1)
        {
            fprintf(stderr, "ERROR: Workspace overflow.\n");
            return ier;
        }
        else if (ier == 2)
        {
            fprintf(stderr, "ERROR: Impossible result! (s too small?)\n");
            fprintf(stderr, "### Reverting to single-shot fit.\n");
            fail = true;
            k = 0;
            iopt = -1;
            continue;
        }
        else if (ier == 3)
        {
            fprintf(stderr, "ERROR: Iteration limit. (s too small?)\n");
            fprintf(stderr, "### Reverting to single-shot fit.\n");
            fail = true;
            k = 0;
            iopt = -1;
            continue;
        }
        else if (ier == 10)
        {
            fprintf(stderr, "ERROR: Invalid input arguments.\n");
            return ier;
        }
        fail = false;

        // Print knot positions.
        printf(" ## Pass %d has knots (nx,ny)=(%d,%d), s=%.4f, fp=%.4f\n",
                k+1, nx, ny, s, fp);
        printf("    x:\n");
        for (int j = 0; j < nx; ++j) printf(" %.3f", tx[j]); printf("\n");
        printf("    y:\n");
        for (int j = 0; j < ny; ++j) printf(" %.3f", ty[j]); printf("\n\n");

        // Reduce smoothness parameter.
        s = s / 1.2;

        // Increment counter.
        ++k;
    } while (fp / ((nx-2*(kxy+1)) * (ny-2*(kxy+1))) > pow(2.0 * noise, 2) &&
            k < 1000 && (iopt >= 0 || fail));
    TIMER_STOP("Finished precalculation");

    // Interpolate.
    int out_x = 701;
    int out_y = 501;
    vector<float> output(out_x * out_y);
    TIMER_START
    int one = 1;
    for (int j = 0, k = 0; j < out_y; ++j)
    {
        float py = y[size_y - 1] * float(j) / (out_y - 1);
        for (int i = 0; i < out_x; ++i, ++k)
        {
            float val;
            float px = x[size_x - 1] * float(i) / (out_x - 1);
            bispev_(&tx[0], &nx, &ty[0], &ny, &c[0], &kxy, &kxy,
                    &px, &one, &py, &one, &val,
                    &wrk[0], &lwrk, &iwrk[0], &kwrk, &ier);
            if (ier != 0)
            {
                fprintf(stderr, "ERROR: Spline evaluation failed (%d)\n", ier);
                return ier;
            }
            output[k] = val;
        }
    }
    TIMER_STOP("Finished interpolation (%d points)", out_x * out_y);

    // Write out the interpolated data.
    file = fopen("test.dat", "w");
    for (int j = 0, k = 0; j < out_y; ++j)
    {
        for (int i = 0; i < out_x; ++i, ++k)
        {
            fprintf(file, "%10.6f ", output[k]);
        }
        fprintf(file, "\n");
    }
    fclose(file);

    return 0;
}
