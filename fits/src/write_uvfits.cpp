#include "fits/write_uvfits.h"
#include <fitsio.h>
#include <cstdlib>
#include <cstdio>

#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QVector>

namespace oskar {

UVFitsWriter::UVFitsWriter()
{
    _fptr = NULL;
    _filename = QString::null;
    _status = 0;
    // Number of decimal places for double keywords.
    _decimals = 10;
    // Number of axes (6 are required, a 7th (BAND axis) is optional).
    _num_axis = 6;
    // Axis dimensions.
    _axis_dim.resize(_num_axis);
    // Number of parameters.
    // 1. UU        (u baseline coordinate)
    // 2. VV        (v baseline coordinate)
    // 3. WW        (w baseline coordinate)
    // 4. DATE      (Julian date)
    // 5. BASELINE  (Baseline number = ant1 * 256 + ant2)
    _num_param = 5;
    _num_vis = 0;
}


UVFitsWriter::~UVFitsWriter()
{
    close_file();
}


void UVFitsWriter::open_file(const QString & filename, const bool replace)
{
    if (QFile::exists(filename) && replace == true)
    {
        printf("UVFitsWriter: Removing existing fits file '%s'\n",
                filename.toLatin1().data());
        QFile::remove(filename);
    }
    _filename = filename;
    // Create and open a new empty output FITS file.
    fits_create_file(&_fptr, filename.toLatin1().data(), &_status);
    check_status("Opening file");
}


void UVFitsWriter::close_file()
{
    if (_fptr != NULL)
    {
        fits_close_file(_fptr, &_status);
        _filename = QString::null;
    }
    check_status();
}


void UVFitsWriter::write_header(const long long num_vis)
{
    // TODO: check if file is open.

    _num_vis = num_vis;

    // Write the required groups header.
    write_groups_header(_num_vis);
    check_status();

    // Extra stuff...
    fits_write_date(_fptr, &_status);
    char key[FLEN_KEYWORD];
    char value[FLEN_VALUE];
    strcpy(key, "TELESCOP");
    strcpy(value, "OSKAR SKA P1");
    fits_write_key_str(_fptr,  key, value, NULL, &_status);
    strcpy(key, "BUNIT");
    strcpy(value, "JY");
    fits_write_key_str(_fptr, key, value, "Units of flux", &_status);
    fits_write_key_dbl(_fptr, "EQUINOX", 2000.0, _decimals,
            "Epoch of RA DEC", &_status);
    fits_write_key_dbl(_fptr, "OBSRA", 0.0, _decimals,
            "Antenna pointing RA", &_status);
    fits_write_key_dbl(_fptr, "OBSDEC", 40.0, _decimals,
            "Antenna pointing DEC", &_status);

    // Axis description headers (NOTE: axis 1 = empty).
    write_axis_header(2, "COMPLEX", "1=real, 2=imag, 3=weight", 1.0, 1.0, 1.0, 1.0);
    write_axis_header(3, "STOKES", "1=I", 1.0, 1.0, 1.0, 1.0);
    write_axis_header(4, "FREQ", "Frequency in Hz.", 600.0e6, 100.0e6, 1.0, 0.0);
    write_axis_header(5, "RA", "Right Ascension in deg.", 0.0, 0.0, 1.0, 1.0);
    write_axis_header(6, "DEC", "Declination in deg.", 90.0, 0.0, 1.0, 1.0);
    check_status();

    // Parameter headers.
    double freq = 600.0e6;
    double invFreq = 1.0 / freq;
    write_param_header(1, "UU--",     "", invFreq, 0.0);
    write_param_header(2, "VV--",     "", invFreq, 0.0);
    write_param_header(3, "WW--",     "", invFreq, 0.0);
    write_param_header(4, "DATE",     "", 1.0,     2451544.5);
    write_param_header(5, "BASELINE", "", 1.0,     0.0);
    check_status();


    // Write a name that is picked up by AIPS.
    char name[FLEN_COMMENT];
    QFileInfo file(_filename);
    QString im_name = "AIPS   IMNAME='" + file.baseName().toUpper() + "'";
    im_name.resize(FLEN_COMMENT - 1);
    strcpy(name, im_name.toLatin1().data());
    fits_write_history(_fptr, name, &_status);
}

void UVFitsWriter::write_groups_header(const long num_stokes,
        const long num_freqs, const long num_ra, const long num_dec)
{
    // TODO: check if file is open.

    int simple = TRUE;          // This file does conform to FITS standard.
    int bitpix = FLOAT_IMG;     // FLOAT_IMG=-32: AIPS dosn't use double!
    int extend = TRUE;          // Allow use of extensions.
                                // Note: TRUE does not require extensions.

    _axis_dim[0] = 0;           // No standard image just group
    _axis_dim[1] = 3;           // (required) real, imaginary, weight.
    _axis_dim[2] = num_stokes;  // (required) Stokes parameters.
    _axis_dim[3] = num_freqs;   // (required) Frequency (spectral channel).
    _axis_dim[4] = num_ra;      // (required) Right ascension of phase centre.
    _axis_dim[5] = num_dec;     // (required) Declination of phase centre.

    long long gcount = _num_vis;     // number of groups (i.e. visibilities)
    long long pcount = _num_param;  // Number of parameters per group.

    // Write random groups description header.
    fits_write_grphdr(_fptr, simple, bitpix, _num_axis, _axis_dim.data(),
            pcount, gcount, extend, &_status);

//    // Remove the two comment lines added at the end of the groups section.
//    fits_delete_key(_fptr, "COMMENT", &_status);
//    fits_delete_key(_fptr, "COMMENT", &_status);

    // Check the CFITSIO error status.
    check_status();
}





void UVFitsWriter::write_axis_header(const int id, const QString & ctype,
        const QString & comment, const double crval, const double cdelt,
        const double crpix, const double crota)
{
    // TODO: check if file is open.

    // Write the axis header.
    char s_key[FLEN_KEYWORD];
    char s_value[FLEN_VALUE];
    char s_comment[FLEN_COMMENT];

    fits_make_keyn("CTYPE", id, s_key, &_status);
    strcpy(s_value, ctype.toLatin1().data());
    strcpy(s_comment, comment.toLatin1().data());
    fits_write_key_str(_fptr, s_key, s_value, s_comment, &_status);

    fits_make_keyn("CRVAL", id, s_key, &_status);
    fits_write_key_dbl(_fptr, s_key, crval, _decimals, NULL, &_status);

    fits_make_keyn("CDELT", id, s_key, &_status);
    fits_write_key_dbl(_fptr, s_key, cdelt, _decimals, NULL, &_status);

    fits_make_keyn("CRPIX", id, s_key, &_status);
    fits_write_key_dbl(_fptr, s_key, crpix, _decimals, NULL, &_status);

    fits_make_keyn("CROTA", id, s_key, &_status);
    fits_write_key_dbl(_fptr, s_key, crota, _decimals, NULL, &_status);
}



void UVFitsWriter::write_param_header(const int id, const QString & type,
        const QString & comment, const double scale, const double zero)
{
    // TODO: check if file is open.

    // Write the parameter header.
    char s_key[FLEN_KEYWORD];
    char s_value[FLEN_VALUE];
    char s_comment[FLEN_COMMENT];

    fits_make_keyn("PTYPE", id, s_key, &_status);
    strcpy(s_value, type.toLatin1().data());
    strcpy(s_comment, comment.toLatin1().data());
    fits_write_key_str(_fptr, s_key, s_value, s_comment, &_status);

    fits_make_keyn("PSCAL", id, s_key, &_status);
    fits_write_key_dbl(_fptr, s_key, scale, _decimals, NULL, &_status);

    fits_make_keyn("PZERO", id, s_key, &_status);
    fits_write_key_dbl(_fptr, s_key, zero, _decimals, NULL, &_status);
}


int UVFitsWriter::num_amps_per_group()
{
    int n = 1;
    for (int i = 1; i < _num_axis; ++i)
        n *= _axis_dim[i];
    return n;
}


void UVFitsWriter::write_data(const float * u, const float * v,
        const float * w, const float * date, const float * baseline,
        const float * re, const float * im, const float * wgt)
{
    // TODO: check file is open for writing...


    // Setup compressed axis dimensions vector.
    QVector<long> naxes(_num_axis - 1);
    for (int i = 0; i < _num_axis - 1; ++i)
        naxes[i] = _axis_dim[i+1];

    QVector<long> fpixel(_num_axis, 1);
    QVector<long> lpixel(naxes);

    int num_values_per_group = num_amps_per_group();
    printf("num values per group = %i\n", num_values_per_group);

    long firstelem = 1;
    long nelements = _num_param;

    QVector<float> p_temp(_num_param);
    QVector<float> g_temp(num_values_per_group);

    for (int i = 0; i < _num_vis; ++i)
    {
        long group = (long)i + 1;

        // Write the parameters.
        p_temp[0] = u[i];
        p_temp[1] = v[i];
        p_temp[2] = w[i];
        p_temp[3] = date[i];
        p_temp[4] = baseline[i];

        printf("- writing group %li\n", group);
        for (int j = 0; j < nelements; ++j)
            printf("   param %i = %f\n", j+1, p_temp[j]);

        fits_write_grppar_flt(_fptr, group, firstelem, nelements,
                p_temp.data(), &_status);
        check_status();

        // Write the data.
        g_temp[0] = re[i];
        g_temp[1] = im[i];
        g_temp[2] = wgt[i];

        for (int j = 0; j < num_values_per_group; ++j)
            printf("   data %i = %f\n", j+1, g_temp[j]);

        fits_write_subset_flt(_fptr, group, naxes.size(), naxes.data(),
                fpixel.data(), lpixel.data(), g_temp.data(), &_status);

        check_status();
    }
}


void UVFitsWriter::check_status(const QString & message)
{
    // No status error, return.
    if (!_status) return;

    fprintf(stderr, "%s\n", QString(80, '*').toLatin1().data());
    // Print user supplied message.
    if (!message.isNull())
        fprintf(stderr, "UVFitsWriter !ERROR!: %s.\n", message.toLatin1().data());

    // Print the CFITSIO error message.
    fits_report_error(stderr, _status);
    fprintf(stderr, "%s\n", QString(80, '*').toLatin1().data());
}

} // namespace oskar


int main(int /*argc*/, char ** /*argv*/)
{
    oskar::UVFitsWriter writer;
    writer.open_file("hello.fits");

    int nvis = 1;
    writer.write_header(nvis);

    int num_amps_per_vis = writer.num_amps_per_group();
    int namps = nvis * num_amps_per_vis;

    QVector<float> u(nvis, 0.0);
    QVector<float> v(nvis, 0.0);
    QVector<float> w(nvis, 0.0);
    QVector<float> date(nvis, 0.0);
    QVector<float> baseline(nvis, 0.0);
    QVector<float> re(namps, 0.0);
    QVector<float> im(namps, 0.0);
    QVector<float> wgt(namps, 0.0);

    baseline[0] = 1.0 * 256.0 + 2.0;
    u[0] = 1050.0;
    re[0] = 1.0;

    writer.write_data(u.constData(), v.constData(), w.constData(), date.constData(),
            baseline.constData(), re.constData(), im.constData(),
            wgt.constData());

    return EXIT_SUCCESS;
}

