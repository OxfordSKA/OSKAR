#include "write_uvfits.h"
#include <fitsio.h>
#include <cstdlib>
#include <cstdio>

#include <QtCore/QFile>
#include <QtCore/QString>
#include <QtCore/QStringList>
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
    int bitpix = FLOAT_IMG; // -32 - AIPS dosn't use double!
    int naxis = 6; //

    int ncomplex = 3; // Re, Im, Weight
    int nstokes = 1;  // I only
    int nfreqs = 1;   // 1 freq. only
    int nra = 1;      // num ra pointings
    int ndec = 1;     // num dec pointings
    QVector<long> naxes(naxis);
    naxes[0] = 0; // No standard image just group
    naxes[1] = ncomplex; // CTYPE2 = Complex (re, im, wgt)
    naxes[2] = nstokes; // CTYPE3 = STOKES (1 = I only)
    naxes[3] = nfreqs; // CTYPE4 = FREQ
    naxes[4] = nra; // CTYPE5 = RA (pointing)
    naxes[5] = ndec; // CTYPE5 = DEC (pointing)
    long long gcount = 1; // number of groups (i.e. visibilities)
    long long pcount = 6; // number of parameters per group.
                          // UU, VV, WW, DATE, DATE, BASELINE
    int extend = FALSE;    // This is the antenna file
    fits_write_grphdr(fptr, simple, bitpix, naxis, naxes.data(), pcount, gcount,
            extend, &status);

    fits_write_date(fptr, &status);

    char * str_key = "TELESCOP";
    char * str_val = "SKA_P1";
    char * str_comment = "";
    fits_write_key_str(fptr, str_key, str_val, str_comment, &status);

    fits_write_key_str(fptr, "BUNIT", "JY", "Units of flux", &status);

    fits_write_key_dbl(fptr, "EQUINOX", 2000.0, decimals,
            "Epoch of RA DEC", &status);

    fits_write_key_dbl(fptr, "OBSRA", 0.0, decimals,
            "Antenna pointing RA", &status);

    fits_write_key_dbl(fptr, "OBSDEC", 40.0, decimals,
            "Antenna pointing DEC", &status);

    // Axis parameter header keys
    QStringList ctype;
    ctype << "COMPLEX" << "STOKES" << "FREQ" << "RA" << "DEC";
    QStringList ctype_comment;
    ctype_comment << "1=real,2=imag,3=weight" << "-1=RR, -2=LL, -3=RL, -4=LR"
            << "Frequency in Hz" << "Right Ascension in deg."
            << "Declination in deg.";
    QList<double> crval;
    crval << 1.0 << -1.0 << 600.0e6 << 0.0 << 90.0;
    QList<double> cdelt;
    cdelt << 1.0 << -1.0 << 100e6 << 0.0 << 0.0;
    QList<double> crpix;
    crpix << 1.0 << 1.0 << 1.0 << 1.0 << 1.0;
    QList<double> crota;
    crota << 0.0 << 0.0 << 0.0 << 0.0 << 0.0;

    QString key;
    char s_key[FLEN_KEYWORD];
    char s_value[FLEN_VALUE];
    char s_comment[FLEN_COMMENT];

    for (int i = 0; i < naxis - 1; ++i)
    {
        strcpy(s_value, ctype[i].toLatin1().data());
        strcpy(s_comment, ctype_comment[i].toLatin1().data());
        strcpy(s_key, QString("CTYPE%1").arg(i+2).toLatin1().data());
        fits_write_key_str(fptr, s_key, s_value, s_comment,
                &status);

        key = "CRVAL" + QString::number(i+2);
        strcpy(s_key, key.toLatin1().data());
        fits_write_key_dbl(fptr, s_key, crval[i], decimals, "", &status);

        key = "CDELT" + QString::number(i+2);
        strcpy(s_key, key.toLatin1().data());
        fits_write_key_dbl(fptr, s_key, cdelt[i], decimals, "", &status);

        key = "CRPIX" + QString::number(i+2);
        strcpy(s_key, key.toLatin1().data());
        fits_write_key_dbl(fptr, s_key, crpix[i], decimals, "", &status);

        key = "CROTA" + QString::number(i+2);
        strcpy(s_key, key.toLatin1().data());
        fits_write_key_dbl(fptr, s_key, crota[i], decimals, "", &status);
    }


    // Group parameter header keys
    QStringList ptype;
    ptype << "UU--" << "VV--" << "WW--" << "DATE" << "DATE" << "BASELINE";
    QList<double> pscal;
    double freq = 600e6;
    double invfreq = 1.0 / freq;
    pscal << invfreq << invfreq << invfreq << 1.0 << 1.0 << 1.0;
    QList<double> pzero;
    pzero << 0.0 << 0.0 << 0.0 << 2451544.5 << 0.0 << 0.0;
    for (int i = 0; i < pcount; ++i)
    {
        key = "PTYPE" + QString::number(i+1);
        strcpy(s_key, key.toLatin1().data());
        strcpy(s_value, ptype[i].toLatin1().data());
        fits_write_key_str(fptr, s_key, s_value, "", &status);

        key = "PSCAL" + QString::number(i+1);
        strcpy(s_key, key.toLatin1().data());
        fits_write_key_dbl(fptr, s_key, pscal[i], decimals, "", &status);

        key = "PZERO" + QString::number(i+1);
        strcpy(s_key, key.toLatin1().data());
        fits_write_key_dbl(fptr, s_key, pzero[i], decimals, "", &status);
    }

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
//
//    // Write a history line.
//    const char * history = "Those who cannot remember the past are condemned "
//            " to repeat it";
//    fits_write_history(fptr, history, &status);

    // Write data
    // ========================================================================


    // Parameters
    QVector<float> u(gcount, 0.0);
    QVector<float> v(gcount, 0.0);
    QVector<float> w(gcount, 0.0);
    QVector<float> date1(gcount, 0.0);
    QVector<float> date2(gcount, 0.0);
    QVector<float> baseline(gcount, 0.0); // 256 * ant1 + ant2

    u[0] = 1000.0;
    baseline[0] = 256.0 * 1.0 + 2.0;

    // Data.
    QVector<float> vis_re(gcount, 0.0);
    QVector<float> vis_im(gcount, 0.0);
    QVector<float> vis_wgt(gcount, 0.0);

    vis_re[0] = 1.0;
    vis_wgt[0] = 1.0;

    //
    QVector<long> naxes2(naxis - 1);
    for (int i = 0; i < naxis - 1; ++i)
    {
        naxes2[i] = naxes[i+1];
        printf("naxes %d = %d\n", i + 1, naxes2[i]);
    }

    QVector<long> fpixel(naxis, 1);
    QVector<long> lpixel(naxes2);

    fpixel[0] = 1; // Complex
    fpixel[0] = 3;

    fpixel[1] = 1; // Stokes
    fpixel[1] = 1;

    fpixel[2] = 1; // Freq
    fpixel[2] = 1;

    fpixel[3] = 1; // Ra
    fpixel[3] = 1;

    fpixel[4] = 1; //Dec
    fpixel[4] = 1;

    int num_values_per_group = 1;
    for (int i = 0; i < naxis - 1; ++i)
        num_values_per_group *= naxes2[i];
    printf("num values per group = %d\n", num_values_per_group);

    long firstelem = 1;
    long nelements = pcount;

    QVector<float> p_temp(pcount);
    QVector<float> g_temp(num_values_per_group);



    for (int i = 0; i < gcount; ++i)
    {
        long group = (long)i + 1;

        // Write the parameters.
        p_temp[0] = u[i];
        p_temp[1] = v[i];
        p_temp[2] = w[i];
        p_temp[3] = date1[i];
        p_temp[4] = date2[i];
        p_temp[5] = baseline[i];
        printf("- writing group %d\n", group);
        for (int j = 0; j < nelements; ++j)
            printf("   param %d = %f\n", j+1, p_temp[j]);

        fits_write_grppar_flt(fptr, group, firstelem, nelements,
                p_temp.data(), &status);
        ffrprt(stderr, status);

        // Write the data.
        g_temp[0] = vis_re[i];
        g_temp[1] = vis_im[i];
        g_temp[2] = vis_wgt[i];
        for (int j = 0; j < num_values_per_group; ++j)
            printf("   data %d = %f\n", j+1, g_temp[j]);

        fits_write_subset_flt(fptr, group, naxis-1, naxes2.data(),
                fpixel.data(), lpixel.data(), g_temp.data(), &status);
        ffrprt(stderr, status);
    }


    // Close the file.
    fits_close_file(fptr, &status);
}


int main(int /*argc*/, char ** /*argv*/)
{
    write_uvfits();

    return EXIT_SUCCESS;
}

