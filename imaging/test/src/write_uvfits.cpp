#include "write_uvfits.h"
#include <fitsio.h>
#include <cstdlib>
#include <cstdio>

#include <QtCore/QFile>
#include <QtCore/QString>
#include <QtCore/QVector>

void write_uvfits()
{
    fitsfile * fptr;
    const char * filename = "test.fits";
    int status = 0;

    // Remove file if already exists as fits_create_file does no overwrite.
    if (QFile::exists(QString(filename)))
    {
        printf("WARNING: removing fits file = '%s'\n", filename);
        QFile::remove(filename);
    }

    // Create the fits file to write to.
    if (fits_create_file(&fptr, filename, &status))
        ffrprt(stderr, status);


    // Write header.
    // ========================================================================
    int decimals = 9; // number of decimals for header values.

    int simple = TRUE;
    int bitpix = -32;
    int naxis = 6; //
    QVector<long> naxes(naxis);
    naxes[0] = 0; // No standard image just group
    naxes[1] = 3; // CTYPE2 = Complex (re, im, wgt)
    naxes[2] = 1; // CTYPE3 = STOKES (1 = I only)
    naxes[3] = 1; // CTYPE4 = FREQ (IF)
    naxes[4] = 1; // CTYPE5 = RA (pointing)
    naxes[5] = 1; // CTYPE5 = DEC (pointing)
    long long gcount = 2; // number of groups (i.e. visibilities)
    long long pcount = 6; // number of parameters per group.
                          // UU, VV, WW, DATE, DATE, BASELINE
    int extend = TRUE;    // This is the antenna file
    fits_write_grphdr(fptr, simple, bitpix, naxis, naxes.data(), pcount, gcount,
            extend, &status);

    char * str_key = "TELESCOP";
    char * str_val = "SKA_P1";
    char * str_comment = "";
    fits_write_key_str(fptr, str_key, str_val, str_comment, &status);

    fits_write_key_str(fptr, "BUNIT", "JY", "Units of flux", &status);

    fits_write_key_dbl(fptr, "EQUINOX", 2000.0, decimals,
            "Epoch of RA DEC", &status);

    fits_write_key_dbl(fptr, "OBSRA", 0.0, decimals,
            "Epoch of RA DEC", &status);

    fits_write_key_dbl(fptr, "OBSDEC", 40.0, decimals,
            "Antenna pointing DEC", &status);


    char * ctype_values[] = {"COMPLEX","STOKES","FREQ","RA","DEC"};
    char * ctype_comments[] = {
            "1=real,2=imag,3=weight",
            "-1=RR, -2=LL, -3=RL, -4=LR",
            "Frequency in Hz",
            "Right Ascension in deg.",
            "Declination in deg."};
    fits_write_keys_str(fptr, "CTYPE", 2, naxis-1, ctype_values, ctype_comments,
            &status);

    QVector<double> cval_values(naxis-1);
    cval_values[0] = 1.0;
    cval_values[1] = 2.0;
    cval_values[2] = 3.0;
    cval_values[3] = 4.0;
    cval_values[4] = 5.0;
    char * cval_comments[] = { "", "", "", "", "" };
    fits_write_keys_dbl(fptr, "CVAL", 2, naxis - 1, cval_values.data(),
            decimals, cval_comments, &status);


//    // Write a single double key.
//    const char * keyname = "TEST";
//    double value = 1.0;
//    const char * comment = "hello!";
//    fits_write_key_dbl(fptr, keyname, value, decimals, comment, &status);
//
//    // EXAMPLE: Write a vector of keys.
//    const char * keyroot = "KEYROOT";
//    int nstart = 1;
//    int nkeys = 3;
//    QVector<double> values(nkeys);
//    values[0] = 1.0;
//    values[1] = 2.0;
//    values[2] = 3.0;
//    char ** comments = NULL;
//    fits_write_keys_dbl(fptr, keyroot, nstart, nkeys, values.data(), decimals,
//            comments, &status);
//
//    fits_write_date(fptr, &status);
//
//    // Write a history line.
//    const char * history = "Those who cannot remember the past are condemned "
//            " to repeat it";
//    fits_write_history(fptr, history, &status);

    // Write data
    // ========================================================================
    // fits_write_grppar_dbl/flt ...


    // Close the file.
    fits_close_file(fptr, &status);
}


int main(int /*argc*/, char ** /*argv*/)
{
    write_uvfits();

    return EXIT_SUCCESS;
}
