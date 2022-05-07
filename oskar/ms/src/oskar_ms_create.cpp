/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "ms/oskar_measurement_set.h"
#include "ms/private_ms.h"

#include <tables/Tables.h>
#include <casa/Arrays/Vector.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace casacore;

static void oskar_ms_add_band(oskar_MeasurementSet* p, int pol_id,
        unsigned int num_channels, double ref_freq,
        const Vector<double>& chan_freqs,
        const Vector<double>& chan_widths);
static void oskar_ms_add_pol(oskar_MeasurementSet* p, unsigned int num_pols);

#ifdef OSKAR_MS_NEW
static void add_column_metadata(TableDesc& desc, const String& column,
    int num_dim, const String& unit, String type = "", String ref = "")
{
    Array<String> units(IPosition(1, num_dim), unit);
    desc.rwColumnDesc(column).rwKeywordSet().define("QuantumUnits", units);
    if (type.empty()) return;
    RecordDesc rec_desc;
    rec_desc.addField("type", TpString);
    rec_desc.addField("Ref", TpString);
    Record rec(rec_desc);
    rec.define("type", type);
    rec.define("Ref", ref);
    desc.rwColumnDesc(column).rwKeywordSet().defineRecord("MEASINFO", rec);
}

static void add_column_frequency(TableDesc& desc, const String& column)
{
    Array<String> units(IPosition(1, 1), String("Hz"));
    desc.rwColumnDesc(column).rwKeywordSet().define("QuantumUnits", units);
    RecordDesc rec_desc;
    rec_desc.addField("type", TpString);
    rec_desc.addField("VarRefCol", TpString);
    rec_desc.addField("TabRefTypes", TpArrayString);
    rec_desc.addField("TabRefCodes", TpArrayUInt);
    Array<String> types(IPosition(1, 10));
    Array<uInt> codes(IPosition(1, 10));
    codes[0] = 0; codes[1] = 1; codes[2] = 2; codes[3] = 3;
    codes[4] = 4; codes[5] = 5; codes[6] = 6; codes[7] = 7;
    codes[8] = 8; codes[9] = 64;
    types[0] = "REST"; types[1] = "LSRK"; types[2] = "LSRD"; types[3] = "BARY";
    types[4] = "GEO";  types[5] = "TOPO"; types[6] = "GALACTO";
    types[7] = "LGROUP"; types[8] = "CMB"; types[9] = "Undefined";
    Record rec(rec_desc);
    rec.define("type", "frequency");
    rec.define("VarRefCol", "MEAS_FREQ_REF");
    rec.define("TabRefTypes", types);
    rec.define("TabRefCodes", codes);
    desc.rwColumnDesc(column).rwKeywordSet().defineRecord("MEASINFO", rec);
}

void oskar_ms_bind_refs(oskar_MeasurementSet* p)
{
    p->msmc.antenna1.attach(*(p->ms), "ANTENNA1");
    p->msmc.antenna2.attach(*(p->ms), "ANTENNA2");
    p->msmc.sigma.attach(*(p->ms), "SIGMA");
    p->msmc.weight.attach(*(p->ms), "WEIGHT");
    p->msmc.exposure.attach(*(p->ms), "EXPOSURE");
    p->msmc.interval.attach(*(p->ms), "INTERVAL");
    p->msmc.time.attach(*(p->ms), "TIME");
    p->msmc.timeCentroid.attach(*(p->ms), "TIME_CENTROID");
    p->msmc.uvw.attach(*(p->ms), "UVW");
    p->msmc.data.attach(*(p->ms), "DATA");
}

static void create_table_antenna(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ArrayColumnDesc<Double>(
            "OFFSET",
            "Axes offset of mount to FEED REFERENCE point", IPosition(1, 3)));
    desc.addColumn(ArrayColumnDesc<Double>(
            "POSITION",
            "Antenna X,Y,Z phase reference position", IPosition(1, 3)));
    desc.addColumn(ScalarColumnDesc<String>(
            "TYPE", "Antenna type (e.g. SPACE-BASED)"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "DISH_DIAMETER", "Physical diameter of dish"));
    desc.addColumn(ScalarColumnDesc<Bool>("FLAG_ROW", "Flag for this row"));
    desc.addColumn(ScalarColumnDesc<String>(
            "MOUNT", "Mount type e.g. alt-az, equatorial, etc."));
    desc.addColumn(ScalarColumnDesc<String>(
            "NAME", "Antenna name, e.g. VLA22, CA03"));
    desc.addColumn(ScalarColumnDesc<String>(
            "STATION", "Station (antenna pad) name"));
    add_column_metadata(desc, "OFFSET", 3, "m", "position", "ITRF");
    add_column_metadata(desc, "POSITION", 3, "m", "position", "ITRF");
    add_column_metadata(desc, "DISH_DIAMETER", 1, "m", "", "");
    SetupNewTable tab(p->ms->tableName() + "/ANTENNA", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("ANTENNA", Table(tab));
}

static void create_table_data_desc(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ScalarColumnDesc<Bool>("FLAG_ROW", "Flag this row"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "POLARIZATION_ID", "Pointer to polarization table"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "SPECTRAL_WINDOW_ID", "Pointer to spectralwindow table"));
    SetupNewTable tab(p->ms->tableName() + "/DATA_DESCRIPTION", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("DATA_DESCRIPTION", Table(tab));
}

static void create_table_feed(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ArrayColumnDesc<Double>(
            "POSITION", "Position of feed relative to feed reference position",
            IPosition(1, 3)));
    desc.addColumn(ArrayColumnDesc<Double>(
            "BEAM_OFFSET",
            "Beam position offset (on sky but in antenna reference frame)"));
    desc.addColumn(ArrayColumnDesc<String>(
            "POLARIZATION_TYPE",
            "Type of polarization to which a given RECEPTOR responds"));
    desc.addColumn(ArrayColumnDesc<Complex>(
            "POL_RESPONSE", "D-matrix i.e. leakage between two receptors"));
    desc.addColumn(ArrayColumnDesc<Double>(
            "RECEPTOR_ANGLE", "The reference angle for polarization"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "ANTENNA_ID", "ID of antenna in this array"));
    desc.addColumn(ScalarColumnDesc<Int>("BEAM_ID", "Id for BEAM model"));
    desc.addColumn(ScalarColumnDesc<Int>("FEED_ID", "Feed id"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "INTERVAL",
            "Interval for which this set of parameters is accurate"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "NUM_RECEPTORS",
            "Number of receptors on this feed (probably 1 or 2)"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "SPECTRAL_WINDOW_ID", "ID for this spectral window setup"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "TIME",
            "Midpoint of time for which this set of parameters is accurate"));
    add_column_metadata(desc, "POSITION", 3, "m", "position", "ITRF");
    add_column_metadata(desc, "BEAM_OFFSET", 2, "rad", "direction", "J2000");
    add_column_metadata(desc, "RECEPTOR_ANGLE", 1, "rad", "", "");
    add_column_metadata(desc, "INTERVAL", 1, "s", "", "");
    add_column_metadata(desc, "TIME", 1, "s", "epoch", "UTC");
    SetupNewTable tab(p->ms->tableName() + "/FEED", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("FEED", Table(tab));
}

static void create_table_flag_cmd(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ScalarColumnDesc<Bool>(
            "APPLIED", "True if flag has been applied to main table"));
    desc.addColumn(ScalarColumnDesc<String>(
            "COMMAND", "Flagging command"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "INTERVAL", "Time interval for which this flag is valid"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "LEVEL", "Flag level - revision level "));
    desc.addColumn(ScalarColumnDesc<String>(
            "REASON", "Flag reason"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "SEVERITY", "Severity code (0-10) "));
    desc.addColumn(ScalarColumnDesc<Double>(
            "TIME", "Midpoint of interval for which this flag is valid"));
    desc.addColumn(ScalarColumnDesc<String>(
            "TYPE", "Type of flag (FLAG or UNFLAG)"));
    add_column_metadata(desc, "INTERVAL", 1, "s", "", "");
    add_column_metadata(desc, "TIME", 1, "s", "epoch", "UTC");
    SetupNewTable tab(p->ms->tableName() + "/FLAG_CMD", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("FLAG_CMD", Table(tab));
}

static void create_table_field(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ArrayColumnDesc<Double>(
            "DELAY_DIR",
            "Direction of delay center (e.g. RA, DEC) as polynomial in time.",
            2));
    desc.addColumn(ArrayColumnDesc<Double>(
            "PHASE_DIR",
            "Direction of phase center (e.g. RA, DEC) as polynomial in time.",
            2));
    desc.addColumn(ArrayColumnDesc<Double>(
            "REFERENCE_DIR",
            "Direction of REFERENCE center (e.g. RA, DEC) as polynomial in time.",
            2));
    desc.addColumn(ScalarColumnDesc<String>(
            "CODE",
            "Special characteristics of field, e.g. Bandpass calibrator"));
    desc.addColumn(ScalarColumnDesc<Bool>("FLAG_ROW", "Row Flag"));
    desc.addColumn(ScalarColumnDesc<String>("NAME", "Name of this field"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "NUM_POLY", "Polynomial order of _DIR columns"));
    desc.addColumn(ScalarColumnDesc<Int>("SOURCE_ID", "Source id"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "TIME", "Time origin for direction and rate"));
    add_column_metadata(desc, "DELAY_DIR", 2, "rad", "direction", "J2000");
    add_column_metadata(desc, "PHASE_DIR", 2, "rad", "direction", "J2000");
    add_column_metadata(desc, "REFERENCE_DIR", 2, "rad", "direction", "J2000");
    add_column_metadata(desc, "TIME", 1, "s", "epoch", "UTC");
    SetupNewTable tab(p->ms->tableName() + "/FIELD", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("FIELD", Table(tab));
}

static void create_table_history(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ArrayColumnDesc<String>(
            "APP_PARAMS", "Application parameters"));
    desc.addColumn(ArrayColumnDesc<String>(
            "CLI_COMMAND", "CLI command sequence"));
    desc.addColumn(ScalarColumnDesc<String>("APPLICATION", "Application name"));
    desc.addColumn(ScalarColumnDesc<String>("MESSAGE", "Log message"));
    desc.addColumn(ScalarColumnDesc<Int>("OBJECT_ID", "Originating ObjectID"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "OBSERVATION_ID", "Observation id (index in OBSERVATION table)"));
    desc.addColumn(ScalarColumnDesc<String>(
            "ORIGIN", "(Source code) origin from which message originated"));
    desc.addColumn(ScalarColumnDesc<String>("PRIORITY", "Message priority"));
    desc.addColumn(ScalarColumnDesc<Double>("TIME", "Timestamp of message"));
    add_column_metadata(desc, "TIME", 1, "s", "epoch", "UTC");
    SetupNewTable tab(p->ms->tableName() + "/HISTORY", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("HISTORY", Table(tab));
}

static void create_table_observation(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ArrayColumnDesc<Double>(
            "TIME_RANGE", "Start and end of observation", IPosition(1, 2)));
    desc.addColumn(ArrayColumnDesc<String>("LOG", "Observing log"));
    desc.addColumn(ArrayColumnDesc<String>("SCHEDULE", "Observing schedule"));
    desc.addColumn(ScalarColumnDesc<Bool>("FLAG_ROW", "Row flag"));
    desc.addColumn(ScalarColumnDesc<String>("OBSERVER", "Name of observer(s)"));
    desc.addColumn(ScalarColumnDesc<String>(
            "PROJECT", "Project identification string"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "RELEASE_DATE", "Release date when data becomes public"));
    desc.addColumn(ScalarColumnDesc<String>(
            "SCHEDULE_TYPE", "Observing schedule type"));
    desc.addColumn(ScalarColumnDesc<String>(
            "TELESCOPE_NAME", "Telescope Name (e.g. WSRT, VLBA)"));
    desc.addColumn(ArrayColumnDesc<Double>(
            "ARRAY_CENTER", "Reference position for array", IPosition(1, 3)));
    add_column_metadata(desc, "TIME_RANGE", 1, "s", "epoch", "UTC");
    add_column_metadata(desc, "RELEASE_DATE", 1, "s", "epoch", "UTC");
    add_column_metadata(desc, "ARRAY_CENTER", 3, "m", "position", "ITRF");
    SetupNewTable tab(p->ms->tableName() + "/OBSERVATION", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("OBSERVATION", Table(tab));
}

static void create_table_phased_array(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ScalarColumnDesc<Int>("ANTENNA_ID", "Antenna ID"));
    desc.addColumn(ArrayColumnDesc<Double>(
            "POSITION",
            "Position of antenna field", IPosition(1, 3)));
    // FIXME(FD) Should this be COORDINATE_AXES or COORDINATE_SYSTEM?
    // Maybe just add both to be safe.
    desc.addColumn(ArrayColumnDesc<Double>(
            "COORDINATE_AXES",
            "Local coordinate system", IPosition(2, 3, 3)));
    desc.addColumn(ArrayColumnDesc<Double>(
            "COORDINATE_SYSTEM",
            "Local coordinate system", IPosition(2, 3, 3)));
    desc.addColumn(ArrayColumnDesc<Double>(
            "ELEMENT_OFFSET", "Offset per element"));
    desc.addColumn(ArrayColumnDesc<Bool>(
            "ELEMENT_FLAG", "Flag of elements in array"));
    add_column_metadata(desc, "POSITION", 3, "m", "position", "ITRF");
    add_column_metadata(desc, "COORDINATE_AXES", 3, "m", "direction", "ITRF");
    add_column_metadata(desc, "COORDINATE_SYSTEM", 3, "m", "direction", "ITRF");
    add_column_metadata(desc, "ELEMENT_OFFSET", 3, "m", "position", "ITRF");
    SetupNewTable tab(p->ms->tableName() + "/PHASED_ARRAY", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("PHASED_ARRAY", Table(tab));
}

static void create_table_pointing(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ArrayColumnDesc<Double>(
            "DIRECTION", "Antenna pointing direction as polynomial in time"));
    desc.addColumn(ScalarColumnDesc<Int>("ANTENNA_ID", "Antenna Id"));
    desc.addColumn(ScalarColumnDesc<Double>("INTERVAL", "Time interval"));
    desc.addColumn(ScalarColumnDesc<String>("NAME", "Pointing position name"));
    desc.addColumn(ScalarColumnDesc<Int>("NUM_POLY", "Series order"));
    desc.addColumn(ArrayColumnDesc<Double>(
            "TARGET", "target direction as polynomial in time"));
    desc.addColumn(ScalarColumnDesc<Double>("TIME", "Time interval midpoint"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "TIME_ORIGIN", "Time origin for direction"));
    desc.addColumn(ScalarColumnDesc<Bool>(
            "TRACKING", "Tracking flag - True if on position"));
    add_column_metadata(desc, "DIRECTION", 2, "rad", "direction", "J2000");
    add_column_metadata(desc, "TARGET", 2, "rad", "direction", "J2000");
    add_column_metadata(desc, "INTERVAL", 1, "s", "", "");
    add_column_metadata(desc, "TIME", 1, "s", "epoch", "UTC");
    add_column_metadata(desc, "TIME_ORIGIN", 1, "s", "epoch", "UTC");
    SetupNewTable tab(p->ms->tableName() + "/POINTING", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("POINTING", Table(tab));
}

static void create_table_polarization(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ArrayColumnDesc<Int>(
            "CORR_TYPE",
            "The polarization type for each correlation product, as a Stokes enum."));
    desc.addColumn(ArrayColumnDesc<Int>(
            "CORR_PRODUCT",
            "Indices describing receptors of feed going into correlation"));
    desc.addColumn(ScalarColumnDesc<Bool>("FLAG_ROW", "Row flag"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "NUM_CORR", "Number of correlation products"));
    SetupNewTable tab(p->ms->tableName() + "/POLARIZATION", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("POLARIZATION", Table(tab));
}

static void create_table_processor(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ScalarColumnDesc<Bool>("FLAG_ROW", "Row flag"));
    desc.addColumn(ScalarColumnDesc<Int>("MODE_ID", "Processor mode id"));
    desc.addColumn(ScalarColumnDesc<String>("TYPE", "Processor type"));
    desc.addColumn(ScalarColumnDesc<Int>("TYPE_ID", "Processor type id"));
    desc.addColumn(ScalarColumnDesc<String>("SUB_TYPE", "Processor sub type"));
    SetupNewTable tab(p->ms->tableName() + "/PROCESSOR", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("PROCESSOR", Table(tab));
}

static void create_table_source(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ArrayColumnDesc<Double>(
            "DIRECTION", "Direction (e.g. RA, DEC).", IPosition(1, 2)));
    desc.addColumn(ArrayColumnDesc<Double>("PROPER_MOTION", "Proper motion",
            IPosition(1, 2)));
    desc.addColumn(ScalarColumnDesc<Int>(
            "CALIBRATION_GROUP",
            "Number of grouping for calibration purpose."));
    desc.addColumn(ScalarColumnDesc<String>(
            "CODE",
            "Special characteristics of source, e.g. Bandpass calibrator"));
    desc.addColumn(ScalarColumnDesc<Double>("INTERVAL", "Time interval"));
    desc.addColumn(ScalarColumnDesc<String>(
            "NAME", "Name of source as given during observations"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "NUM_LINES", "Number of spectral lines"));
    desc.addColumn(ScalarColumnDesc<Int>("SOURCE_ID", "Source id"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "SPECTRAL_WINDOW_ID", "ID for this spectral window setup"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "TIME",
            "Midpoint of time for which this set of parameters is accurate."));
    desc.addColumn(ArrayColumnDesc<Double>(
            "REST_FREQUENCY", "Line rest frequency"));
    desc.addColumn(ArrayColumnDesc<Double>(
            "POSITION", "Position (e.g. for solar system objects",
            IPosition(1, 3)));
    add_column_metadata(desc, "DIRECTION", 2, "rad", "direction", "J2000");
    add_column_metadata(desc, "PROPER_MOTION", 1, "rad/s", "", "");
    add_column_metadata(desc, "INTERVAL", 1, "s", "", "");
    add_column_metadata(desc, "TIME", 1, "s", "epoch", "UTC");
    add_column_metadata(desc, "REST_FREQUENCY", 1, "Hz", "frequency", "LSRK");
    add_column_metadata(desc, "POSITION", 3, "m", "position", "ITRF");
    SetupNewTable tab(p->ms->tableName() + "/SOURCE", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("SOURCE", Table(tab));
}

static void create_table_spectral_window(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ScalarColumnDesc<Int>(
            "MEAS_FREQ_REF", "Frequency Measure reference"));
    desc.addColumn(ArrayColumnDesc<Double>(
            "CHAN_FREQ",
            "Center frequencies for each channel in the data matrix"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "REF_FREQUENCY", "The reference frequency"));
    desc.addColumn(ArrayColumnDesc<Double>(
            "CHAN_WIDTH", "Channel width for each channel"));
    desc.addColumn(ArrayColumnDesc<Double>(
            "EFFECTIVE_BW", "Effective noise bandwidth of each channel"));
    desc.addColumn(ArrayColumnDesc<Double>(
            "RESOLUTION", "The effective noise bandwidth for each channel"));
    desc.addColumn(ScalarColumnDesc<Bool>("FLAG_ROW", "Row flag"));
    desc.addColumn(ScalarColumnDesc<Int>("FREQ_GROUP", "Frequency group"));
    desc.addColumn(ScalarColumnDesc<String>(
            "FREQ_GROUP_NAME", "Frequency group name"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "IF_CONV_CHAIN", "The IF conversion chain number"));
    desc.addColumn(ScalarColumnDesc<String>("NAME", "Spectral window name"));
    desc.addColumn(ScalarColumnDesc<Int>("NET_SIDEBAND", "Net sideband"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "NUM_CHAN", "Number of spectral channels"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "TOTAL_BANDWIDTH", "The total bandwidth for this window"));
    add_column_frequency(desc, "CHAN_FREQ");
    add_column_frequency(desc, "REF_FREQUENCY");
    add_column_metadata(desc, "CHAN_WIDTH", 1, "Hz", "", "");
    add_column_metadata(desc, "EFFECTIVE_BW", 1, "Hz", "", "");
    add_column_metadata(desc, "RESOLUTION", 1, "Hz", "", "");
    add_column_metadata(desc, "TOTAL_BANDWIDTH", 1, "Hz", "", "");
    SetupNewTable tab(p->ms->tableName() + "/SPECTRAL_WINDOW", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("SPECTRAL_WINDOW", Table(tab));
}

static void create_table_state(oskar_MeasurementSet* p)
{
    TableDesc desc;
    desc.addColumn(ScalarColumnDesc<Double>(
            "CAL", "Noise calibration temperature"));
    desc.addColumn(ScalarColumnDesc<Bool>("FLAG_ROW", "Row flag"));
    desc.addColumn(ScalarColumnDesc<Double>("LOAD", "Load temperature"));
    desc.addColumn(ScalarColumnDesc<String>(
            "OBS_MODE", "Observing mode, e.g., OFF_SPECTRUM"));
    desc.addColumn(ScalarColumnDesc<Bool>(
            "REF", "True for a reference observation"));
    desc.addColumn(ScalarColumnDesc<Bool>(
            "SIG", "True for a source observation"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "SUB_SCAN", "Sub scan number, relative to scan number"));
    add_column_metadata(desc, "CAL", 1, "K", "", "");
    add_column_metadata(desc, "LOAD", 1, "K", "", "");
    SetupNewTable tab(p->ms->tableName() + "/STATE", desc, Table::New);
    p->ms->rwKeywordSet().defineTable("STATE", Table(tab));
}

static void create_subtables(oskar_MeasurementSet* p)
{
    create_table_antenna(p);
    create_table_data_desc(p);
    create_table_feed(p);
    create_table_flag_cmd(p);
    create_table_field(p);
    create_table_history(p);
    create_table_observation(p);
    create_table_phased_array(p);
    create_table_pointing(p);
    create_table_polarization(p);
    create_table_processor(p);
    create_table_source(p);
    create_table_spectral_window(p);
    create_table_state(p);
}
#endif


oskar_MeasurementSet* oskar_ms_create(const char* file_name,
        const char* app_name, unsigned int num_stations,
        unsigned int num_channels, unsigned int num_pols, double freq_start_hz,
        double freq_inc_hz, int write_autocorr, int write_crosscorr)
{
    oskar_MeasurementSet* p = (oskar_MeasurementSet*)
            calloc(1, sizeof(oskar_MeasurementSet));

#ifdef OSKAR_MS_NEW
    // Build the main table descriptor.
    TableDesc desc;
    desc.rwKeywordSet().define("MS_VERSION", Float(2.0));
    desc.addColumn(ArrayColumnDesc<Double>(
            "UVW", "Vector with uvw coordinates (in meters)", IPosition(1, 3)));
    desc.addColumn(ArrayColumnDesc<Bool>(
            "FLAG", // (2D: pol, chan).
            "The data flags, array of bools with same shape as data", 2));
    desc.addColumn(ArrayColumnDesc<Bool>(
            "FLAG_CATEGORY",
            "The flag category, NUM_CAT flags for each datum", 1));
    desc.addColumn(ArrayColumnDesc<Float>(
            "WEIGHT", // (1D: pol).
            "Weight for each polarization spectrum", 1));
    desc.addColumn(ArrayColumnDesc<Float>(
            "SIGMA", // (1D: pol).
            "Estimated rms noise for channel with unity bandpass response", 1));
    desc.addColumn(ScalarColumnDesc<Int>(
            "ANTENNA1", "ID of first antenna in interferometer"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "ANTENNA2", "ID of second antenna in interferometer"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "ARRAY_ID", "ID of array or subarray"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "DATA_DESC_ID", "The data description table index"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "EXPOSURE", "The effective integration time"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "FEED1", "The feed index for ANTENNA1"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "FEED2", "The feed index for ANTENNA2"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "FIELD_ID", "Unique id for this pointing"));
    desc.addColumn(ScalarColumnDesc<Bool>(
            "FLAG_ROW", "Row flag - flag all data in this row if True"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "INTERVAL", "The sampling interval"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "OBSERVATION_ID",
            "ID for this observation, index in OBSERVATION table"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "PROCESSOR_ID",
            "Id for backend processor, index in PROCESSOR table"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "SCAN_NUMBER", "Sequential scan number from on-line system"));
    desc.addColumn(ScalarColumnDesc<Int>(
            "STATE_ID", "ID for this observing state"));
    desc.addColumn(ScalarColumnDesc<Double>(
            "TIME", "Modified Julian Day")); // Comments here are required!
    desc.addColumn(ScalarColumnDesc<Double>(
            "TIME_CENTROID", "Modified Julian Day"));
    desc.addColumn(ArrayColumnDesc<Complex>(
            "DATA", // (2D: pol, chan).
            "The data column", 2));

    // Add column metadata.
    add_column_metadata(desc, "UVW", 3, "m", "uvw", "ITRF");
    desc.rwColumnDesc("FLAG_CATEGORY").rwKeywordSet().define(
            "CATEGORY", Array<String>(IPosition(1, 0), String("")));
    add_column_metadata(desc, "EXPOSURE", 1, "s", "", "");
    add_column_metadata(desc, "INTERVAL", 1, "s", "", "");
    add_column_metadata(desc, "TIME", 1, "s", "epoch", "UTC");
    add_column_metadata(desc, "TIME_CENTROID", 1, "s", "epoch", "UTC");
    desc.rwColumnDesc("DATA").rwKeywordSet().define("UNIT", "Jy");

    // Define column shapes.
    IPosition dataShape(2, num_pols, num_channels);
    IPosition weightShape(1, num_pols);
    //desc.rwColumnDesc("FLAG_CATEGORY").setShape(IPosition(1, 3));
    desc.rwColumnDesc("DATA").setShape(dataShape);
    desc.rwColumnDesc("FLAG").setShape(dataShape);
    desc.rwColumnDesc("WEIGHT").setShape(weightShape);
    desc.rwColumnDesc("SIGMA").setShape(weightShape);
    Vector<String> tsmNames(1);
    tsmNames[0] = "DATA";
    desc.defineHypercolumn("TiledData", 3, tsmNames);
    tsmNames[0] = "FLAG";
    desc.defineHypercolumn("TiledFlag", 3, tsmNames);
    tsmNames[0] = "UVW";
    desc.defineHypercolumn("TiledUVW", 2, tsmNames);
    tsmNames[0] = "WEIGHT";
    desc.defineHypercolumn("TiledWeight", 2, tsmNames);
    tsmNames[0] = "SIGMA";
    desc.defineHypercolumn("TiledSigma", 2, tsmNames);
    try
    {
        unsigned int num_baselines = 0;

        if (write_autocorr && write_crosscorr)
        {
            num_baselines = num_stations * (num_stations + 1) / 2;
        }
        else if (!write_autocorr && write_crosscorr)
        {
            num_baselines = num_stations * (num_stations - 1) / 2;
        }
        else if (write_autocorr && !write_crosscorr)
        {
            num_baselines = num_stations;
        }
        else
        {
            oskar_ms_close(p);
            return 0;
        }

        SetupNewTable tab(file_name, desc, Table::New);

        // Create the default storage managers.
        IncrementalStMan incrStorageManager("ISMData");
        tab.bindAll(incrStorageManager);
        StandardStMan stdStorageManager("SSMData", 32768, 32768);
        tab.bindColumn("ANTENNA1", stdStorageManager);
        tab.bindColumn("ANTENNA2", stdStorageManager);

        // Create tiled column storage manager for UVW column.
        IPosition uvwTileShape(2, 3, 2 * num_baselines);
        TiledColumnStMan uvwStorageManager("TiledUVW", uvwTileShape);
        tab.bindColumn("UVW", uvwStorageManager);

        // Create tiled column storage managers for WEIGHT and SIGMA columns.
        IPosition weightTileShape(2, num_pols, 2 * num_baselines);
        TiledColumnStMan weightStorageManager("TiledWeight", weightTileShape);
        tab.bindColumn("WEIGHT", weightStorageManager);
        IPosition sigmaTileShape(2, num_pols, 2 * num_baselines);
        TiledColumnStMan sigmaStorageManager("TiledSigma", sigmaTileShape);
        tab.bindColumn("SIGMA", sigmaStorageManager);

        // Create tiled column storage managers for DATA and FLAG columns.
        const int tile_channels = num_channels > 8 ? 8 : num_channels;
        IPosition dataTileShape(3, num_pols, tile_channels, 2 * num_baselines);
        TiledColumnStMan dataStorageManager("TiledData", dataTileShape);
        tab.bindColumn("DATA", dataStorageManager);
        IPosition flagTileShape(3, num_pols, tile_channels, 16 * num_baselines);
        TiledColumnStMan flagStorageManager("TiledFlag", flagTileShape);
        tab.bindColumn("FLAG", flagStorageManager);

        // Create the main table and subtables.
        p->ms = new Table(tab, TableLock(TableLock::PermanentLocking));
        create_subtables(p);

        // Bind main table column references.
        oskar_ms_bind_refs(p);
        const size_t app_name_len = 1 + strlen(app_name);
        free(p->app_name);
        p->app_name = (char*) calloc(app_name_len, sizeof(char));
        if (p->app_name) memcpy(p->app_name, app_name, app_name_len);
    }
    catch (AipsError& e)
    {
        fprintf(stderr, "Caught AipsError: %s\n", e.what());
        fflush(stderr);
        oskar_ms_close(p);
        return 0;
    }
    catch (...)
    {
        fprintf(stderr, "Unknown error creating Measurement Set!\n");
        oskar_ms_close(p);
        return 0;
    }

    // Add a row to the OBSERVATION subtable.
    const char* username = getenv("USERNAME");
    if (!username)
    {
        username = getenv("USER");
    }
    p->start_time = DBL_MAX;
    p->end_time = -DBL_MAX;
    Vector<String> corrSchedule(1);
    Vector<Double> timeRangeVal(2, 0.0);
    Table obs(p->ms->tableName() + "/OBSERVATION", Table::Update);
    obs.addRow();
    ArrayColumn<String> schedule(obs, "SCHEDULE");
    ScalarColumn<String> project(obs, "PROJECT"), observer(obs, "OBSERVER");
    ScalarColumn<String> telescopeName(obs, "TELESCOPE_NAME");
    ArrayColumn<Double> timeRange(obs, "TIME_RANGE");
    schedule.put(0, corrSchedule);
    project.put(0, "");
    observer.put(0, username ? username : "Unknown");
    telescopeName.put(0, app_name);
    timeRange.put(0, timeRangeVal);
    oskar_ms_set_time_range(p);

    // Add polarisation ID.
    oskar_ms_add_pol(p, num_pols);

    // Add a dummy field to size the FIELD table.
    oskar_ms_set_phase_centre(p, 0, 0.0, 0.0);

    // Set up the band.
    Vector<double> chan_widths(num_channels, freq_inc_hz);
    Vector<double> chan_freqs(num_channels);
    //double start = ref_freq - (num_channels - 1) * chan_width / 2.0;
    for (unsigned int c = 0; c < num_channels; ++c)
    {
        chan_freqs(c) = freq_start_hz + c * freq_inc_hz;
    }
    oskar_ms_add_band(p, 0, num_channels, freq_start_hz, chan_freqs, chan_widths);

    // Get a string containing the current system time.
    char time_str[80];
    time_t unix_time = 0;
    unix_time = std::time(NULL);
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d, %H:%M:%S (%Z)",
            std::localtime(&unix_time));

    // Add a row to the HISTORY subtable.
    String msg = String("Measurement Set created at ") + String(time_str);
    oskar_ms_add_history(p, app_name, msg.c_str(), msg.size());

    // Set the private data.
    p->num_pols = num_pols;
    p->num_channels = num_channels;
    p->num_stations = num_stations;
    p->num_receptors = 2; // By default.
    p->freq_start_hz = freq_start_hz;
    p->freq_inc_hz = freq_inc_hz;

    // Fill the ANTENNA table.
    Vector<Double> pos(3, 0.0), off(3, 0.0);
    Table antenna(p->ms->tableName() + "/ANTENNA", Table::Update);
    antenna.addRow(num_stations);
    ArrayColumn<Double> position(antenna, "POSITION");
    ArrayColumn<Double> offset(antenna, "OFFSET");
    ScalarColumn<String> mount(antenna, "MOUNT");
    ScalarColumn<String> name(antenna, "NAME"), station(antenna, "STATION");
    ScalarColumn<Double> dishDiameter(antenna, "DISH_DIAMETER");
    ScalarColumn<Bool> flagRow(antenna, "FLAG_ROW");
    for (unsigned int a = 0; a < num_stations; ++a)
    {
        std::ostringstream output;
        output << "s" << std::setw(4) << std::setfill('0') << a;
        String label(output.str());
        offset.put(a, off);
        position.put(a, pos);
        mount.put(a, "FIXED");
        name.put(a, label);
        station.put(a, label);
        dishDiameter.put(a, 1);
        flagRow.put(a, false);
    }

    // Fill the FEED table.
    Matrix<Double> feedOffset(2, p->num_receptors, 0.0);
    Matrix<Complex> feedResponse(p->num_receptors, p->num_receptors,
            Complex(0.0, 0.0));
    Vector<String> feedType(p->num_receptors);
    feedType(0) = "X";
    if (p->num_receptors > 1) feedType(1) = "Y";
    Vector<Double> feedAngle(p->num_receptors, 0.0);
    Table feed(p->ms->tableName() + "/FEED", Table::Update);
    feed.addRow(num_stations);
    ScalarColumn<Int> antennaId(feed, "ANTENNA_ID");
    ArrayColumn<Double> beamOffset(feed, "BEAM_OFFSET");
    ArrayColumn<String> polarizationType(feed, "POLARIZATION_TYPE");
    ArrayColumn<Complex> polResponse(feed, "POL_RESPONSE");
    ArrayColumn<Double> receptorAngle(feed, "RECEPTOR_ANGLE");
    ScalarColumn<Int> numReceptors(feed, "NUM_RECEPTORS");
    for (unsigned int a = 0; a < num_stations; ++a)
    {
        antennaId.put(a, a);
        beamOffset.put(a, feedOffset);
        polarizationType.put(a, feedType);
        polResponse.put(a, feedResponse);
        receptorAngle.put(a, feedAngle);
        numReceptors.put(a, p->num_receptors);
    }

#else
    // Create the table descriptor and use it to set up a new main table.
    TableDesc desc = MS::requiredTableDesc();
    MS::addColumnToDesc(desc, MS::DATA, 2); // Visibilities (2D: pol, chan).
    desc.rwColumnDesc(MS::columnName(MS::DATA)).
            rwKeywordSet().define("UNIT", "Jy");
    IPosition dataShape(2, num_pols, num_channels);
    IPosition weightShape(1, num_pols);
    desc.rwColumnDesc(MS::columnName(MS::DATA)).setShape(dataShape);
    desc.rwColumnDesc(MS::columnName(MS::FLAG)).setShape(dataShape);
    desc.rwColumnDesc(MS::columnName(MS::WEIGHT)).setShape(weightShape);
    desc.rwColumnDesc(MS::columnName(MS::SIGMA)).setShape(weightShape);
    Vector<String> tsmNames(1);
    tsmNames[0] = MS::columnName(MS::DATA);
    desc.defineHypercolumn("TiledData", 3, tsmNames);
    tsmNames[0] = MS::columnName(MS::FLAG);
    desc.defineHypercolumn("TiledFlag", 3, tsmNames);
    tsmNames[0] = MS::columnName(MS::UVW);
    desc.defineHypercolumn("TiledUVW", 2, tsmNames);
    tsmNames[0] = MS::columnName(MS::WEIGHT);
    desc.defineHypercolumn("TiledWeight", 2, tsmNames);
    tsmNames[0] = MS::columnName(MS::SIGMA);
    desc.defineHypercolumn("TiledSigma", 2, tsmNames);
    try
    {
        unsigned int num_baselines = 0;

        if (write_autocorr && write_crosscorr)
            num_baselines = num_stations * (num_stations + 1) / 2;
        else if (!write_autocorr && write_crosscorr)
            num_baselines = num_stations * (num_stations - 1) / 2;
        else if (write_autocorr && !write_crosscorr)
            num_baselines = num_stations;
        else
        {
            oskar_ms_close(p);
            return 0;
        }

        SetupNewTable tab(file_name, desc, Table::New);

        // Create the default storage managers.
        IncrementalStMan incrStorageManager("ISMData");
        tab.bindAll(incrStorageManager);
        StandardStMan stdStorageManager("SSMData", 32768, 32768);
        tab.bindColumn(MS::columnName(MS::ANTENNA1), stdStorageManager);
        tab.bindColumn(MS::columnName(MS::ANTENNA2), stdStorageManager);

        // Create tiled column storage manager for UVW column.
        IPosition uvwTileShape(2, 3, 2 * num_baselines);
        TiledColumnStMan uvwStorageManager("TiledUVW", uvwTileShape);
        tab.bindColumn(MS::columnName(MS::UVW), uvwStorageManager);

        // Create tiled column storage managers for WEIGHT and SIGMA columns.
        IPosition weightTileShape(2, num_pols, 2 * num_baselines);
        TiledColumnStMan weightStorageManager("TiledWeight", weightTileShape);
        tab.bindColumn(MS::columnName(MS::WEIGHT), weightStorageManager);
        IPosition sigmaTileShape(2, num_pols, 2 * num_baselines);
        TiledColumnStMan sigmaStorageManager("TiledSigma", sigmaTileShape);
        tab.bindColumn(MS::columnName(MS::SIGMA), sigmaStorageManager);

        // Create tiled column storage managers for DATA and FLAG columns.
        IPosition dataTileShape(3, num_pols, num_channels, 2 * num_baselines);
        TiledColumnStMan dataStorageManager("TiledData", dataTileShape);
        tab.bindColumn(MS::columnName(MS::DATA), dataStorageManager);
        IPosition flagTileShape(3, num_pols, num_channels, 16 * num_baselines);
        TiledColumnStMan flagStorageManager("TiledFlag", flagTileShape);
        tab.bindColumn(MS::columnName(MS::FLAG), flagStorageManager);

        // Create the Measurement Set.
        p->ms = new MeasurementSet(tab, TableLock(TableLock::PermanentLocking));

        // Create SOURCE sub-table.
        TableDesc descSource = MSSource::requiredTableDesc();
        MSSource::addColumnToDesc(descSource, MSSource::REST_FREQUENCY);
        MSSource::addColumnToDesc(descSource, MSSource::POSITION);
        SetupNewTable sourceSetup(p->ms->sourceTableName(),
                descSource, Table::New);
        p->ms->rwKeywordSet().defineTable(MS::keywordName(MS::SOURCE),
                Table(sourceSetup));

        // Create all required default subtables.
        p->ms->createDefaultSubtables(Table::New);

        // Create the MSMainColumns and MSColumns objects for accessing data
        // in the main table and subtables.
        p->msc = new MSColumns(*(p->ms));
        p->msmc = new MSMainColumns(*(p->ms));
        p->app_name = (char*) realloc(p->app_name, strlen(app_name) + 1);
        strcpy(p->app_name, app_name);
    }
    catch (...)
    {
        fprintf(stderr, "Error creating Measurement Set!\n");
        oskar_ms_close(p);
        return 0;
    }

    // Add a row to the OBSERVATION subtable.
    const char* username = getenv("USERNAME");
    if (!username)
        username = getenv("USER");
    p->start_time = DBL_MAX;
    p->end_time = -DBL_MAX;
    p->ms->observation().addRow();
    Vector<String> corrSchedule(1);
    Vector<Double> timeRange(2, 0.0);
    p->msc->observation().schedule().put(0, corrSchedule);
    p->msc->observation().project().put(0, "");
    if (username)
        p->msc->observation().observer().put(0, username);
    else
        p->msc->observation().observer().put(0, "Unknown");
    p->msc->observation().telescopeName().put(0, app_name);
    p->msc->observation().timeRange().put(0, timeRange);
    oskar_ms_set_time_range(p);

    // Add polarisation ID.
    oskar_ms_add_pol(p, num_pols);

    // Add a dummy field to size the FIELD table.
    oskar_ms_set_phase_centre(p, 0, 0.0, 0.0);

    // Set up the band.
    Vector<double> chan_widths(num_channels, freq_inc_hz);
    Vector<double> chan_freqs(num_channels);
    //double start = ref_freq - (num_channels - 1) * chan_width / 2.0;
    for (unsigned int c = 0; c < num_channels; ++c)
    {
        chan_freqs(c) = freq_start_hz + c * freq_inc_hz;
    }
    oskar_ms_add_band(p, 0, num_channels, freq_start_hz, chan_freqs, chan_widths);

    // Get a string containing the current system time.
    char time_str[80];
    time_t unix_time;
    unix_time = std::time(NULL);
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d, %H:%M:%S (%Z)",
            std::localtime(&unix_time));

    // Add a row to the HISTORY subtable.
    String msg = String("Measurement Set created at ") + String(time_str);
    oskar_ms_add_history(p, app_name, msg.c_str(), msg.size());

    // Set the private data.
    p->num_pols = num_pols;
    p->num_channels = num_channels;
    p->num_stations = num_stations;
    p->num_receptors = 2; // By default.
    p->freq_start_hz = freq_start_hz;
    p->freq_inc_hz = freq_inc_hz;

    // Fill the ANTENNA table.
    p->ms->antenna().addRow(num_stations);
    Vector<Double> pos(3, 0.0);
    for (unsigned int a = 0; a < num_stations; ++a)
    {
        std::ostringstream output;
        output << "s" << std::setw(4) << std::setfill('0') << a;
        String label(output.str());
        p->msc->antenna().position().put(a, pos);
        p->msc->antenna().mount().put(a, "FIXED");
        p->msc->antenna().name().put(a, label);
        p->msc->antenna().station().put(a, label);
        p->msc->antenna().dishDiameter().put(a, 1);
        p->msc->antenna().flagRow().put(a, false);
    }

    // Fill the FEED table.
    Matrix<Double> feedOffset(2, p->num_receptors, 0.0);
    Matrix<Complex> feedResponse(p->num_receptors, p->num_receptors,
            Complex(0.0, 0.0));
    Vector<String> feedType(p->num_receptors);
    feedType(0) = "X";
    if (p->num_receptors > 1) feedType(1) = "Y";
    Vector<Double> feedAngle(p->num_receptors, 0.0);
    p->ms->feed().addRow(num_stations);
    for (unsigned int a = 0; a < num_stations; ++a)
    {
        p->msc->feed().antennaId().put(a, a);
        p->msc->feed().beamOffset().put(a, feedOffset);
        p->msc->feed().polarizationType().put(a, feedType);
        p->msc->feed().polResponse().put(a, feedResponse);
        p->msc->feed().receptorAngle().put(a, feedAngle);
        p->msc->feed().numReceptors().put(a, p->num_receptors);
    }
#endif

    return p;
}

void oskar_ms_add_band(oskar_MeasurementSet* p, int pol_id,
        unsigned int num_channels, double ref_freq,
        const Vector<double>& chan_freqs,
        const Vector<double>& chan_widths)
{
    if (!p->ms) return;
#ifdef OSKAR_MS_NEW
    unsigned int row = 0;

    // Add a row to the DATA_DESCRIPTION subtable.
    {
        Table dataDesc(p->ms->tableName() + "/DATA_DESCRIPTION", Table::Update);
        row = dataDesc.nrow();
        dataDesc.addRow();
        ScalarColumn<Int> spectralWindowId(dataDesc, "SPECTRAL_WINDOW_ID");
        ScalarColumn<Int> polarizationId(dataDesc, "POLARIZATION_ID");
        ScalarColumn<Bool> flagRow(dataDesc, "FLAG_ROW");
        spectralWindowId.put(row, row);
        polarizationId.put(row, pol_id);
        flagRow.put(row, false);
    }

    // Get total bandwidth from maximum and minimum.
    Vector<double> start_freqs = chan_freqs - chan_widths / 2.0;
    Vector<double> end_freqs = chan_freqs + chan_widths / 2.0;
    double total_bandwidth = max(end_freqs) - min(start_freqs);

    // Add a row to the SPECTRAL_WINDOW sub-table.
    {
        Table spw(p->ms->tableName() + "/SPECTRAL_WINDOW", Table::Update);
        spw.addRow();
        ScalarColumn<Int> measFreqRef(spw, "MEAS_FREQ_REF");
        ArrayColumn<Double> chanFreq(spw, "CHAN_FREQ");
        ScalarColumn<Double> refFrequency(spw, "REF_FREQUENCY");
        ArrayColumn<Double> chanWidth(spw, "CHAN_WIDTH");
        ArrayColumn<Double> effectiveBW(spw, "EFFECTIVE_BW");
        ArrayColumn<Double> resolution(spw, "RESOLUTION");
        ScalarColumn<Bool> flagRow(spw, "FLAG_ROW");
        ScalarColumn<Int> freqGroup(spw, "FREQ_GROUP");
        ScalarColumn<String> freqGroupName(spw, "FREQ_GROUP_NAME");
        ScalarColumn<Int> ifConvChain(spw, "IF_CONV_CHAIN");
        ScalarColumn<String> name(spw, "NAME");
        ScalarColumn<Int> netSideband(spw, "NET_SIDEBAND");
        ScalarColumn<Int> numChan(spw, "NUM_CHAN");
        ScalarColumn<Double> totalBandwidth(spw, "TOTAL_BANDWIDTH");
        measFreqRef.put(row, 5); // MFrequency::TOPO
        chanFreq.put(row, chan_freqs);
        refFrequency.put(row, ref_freq);
        chanWidth.put(row, chan_widths);
        effectiveBW.put(row, chan_widths);
        resolution.put(row, chan_widths);
        flagRow.put(row, false);
        freqGroup.put(row, 0);
        freqGroupName.put(row, "");
        ifConvChain.put(row, 0);
        name.put(row, "");
        netSideband.put(row, 0);
        numChan.put(row, num_channels);
        totalBandwidth.put(row, total_bandwidth);
    }
#else
    // Add a row to the DATA_DESCRIPTION subtable.
    unsigned int row = p->ms->dataDescription().nrow();
    p->ms->dataDescription().addRow();
    p->msc->dataDescription().spectralWindowId().put(row, row);
    p->msc->dataDescription().polarizationId().put(row, pol_id);
    p->msc->dataDescription().flagRow().put(row, false);

    // Get total bandwidth from maximum and minimum.
    Vector<double> startFreqs = chan_freqs - chan_widths / 2.0;
    Vector<double> endFreqs = chan_freqs + chan_widths / 2.0;
    double totalBandwidth = max(endFreqs) - min(startFreqs);

    // Add a row to the SPECTRAL_WINDOW sub-table.
    p->ms->spectralWindow().addRow();
    MSSpWindowColumns& s = p->msc->spectralWindow();
    s.measFreqRef().put(row, MFrequency::TOPO);
    s.chanFreq().put(row, chan_freqs);
    s.refFrequency().put(row, ref_freq);
    s.chanWidth().put(row, chan_widths);
    s.effectiveBW().put(row, chan_widths);
    s.resolution().put(row, chan_widths);
    s.flagRow().put(row, false);
    s.freqGroup().put(row, 0);
    s.freqGroupName().put(row, "");
    s.ifConvChain().put(row, 0);
    s.name().put(row, "");
    s.netSideband().put(row, 0);
    s.numChan().put(row, num_channels);
    s.totalBandwidth().put(row, totalBandwidth);
#endif
}

#ifdef OSKAR_MS_NEW
static int receptor1(int stokes_type)
{
    const int rec1 = (stokes_type - 1) % 4;
    return (rec1 < 2) ? 0 : 1;
}

static int receptor2(int stokes_type)
{
    const int rec2 = (stokes_type - 1) % 4;
    return (rec2 == 0 || rec2 == 2) ? 0 : 1;
}
#endif

void oskar_ms_add_pol(oskar_MeasurementSet* p, unsigned int num_pols)
{
    if (!p->ms) return;

    // Set up the correlation type, based on number of polarisations.
    Vector<Int> corr_type(num_pols);
#ifdef OSKAR_MS_NEW
    // Note that Stokes::XX, Stokes::XY, Stokes::YX and Stokes::YY
    // are mapped to the integers 9, 10, 11, 12.
    corr_type(0) = 9; // Stokes::XX; Can't be Stokes I if num_pols = 1! (Throws exception.)
    if (num_pols == 2)
    {
        corr_type(1) = 12; // Stokes::YY;
    }
    else if (num_pols == 4)
    {
        corr_type(1) = 10; // Stokes::XY;
        corr_type(2) = 11; // Stokes::YX;
        corr_type(3) = 12; // Stokes::YY;
    }

    // Set up the correlation product, based on number of polarisations.
    Matrix<Int> corr_product(2, num_pols);
    for (unsigned int i = 0; i < num_pols; ++i)
    {
        corr_product(0, i) = receptor1(corr_type(i));
        corr_product(1, i) = receptor2(corr_type(i));
    }

    // Create a new row, and fill the columns.
    Table pol(p->ms->tableName() + "/POLARIZATION", Table::Update);
    unsigned int row = pol.nrow();
    pol.addRow();
    ArrayColumn<Int> corrType(pol, "CORR_TYPE");
    ArrayColumn<Int> corrProduct(pol, "CORR_PRODUCT");
    ScalarColumn<Int> numCorr(pol, "NUM_CORR");
    corrType.put(row, corr_type);
    corrProduct.put(row, corr_product);
    numCorr.put(row, num_pols);
#else
    corr_type(0) = Stokes::XX; // Can't be Stokes I if num_pols = 1! (Throws exception.)
    if (num_pols == 2)
    {
        corr_type(1) = Stokes::YY;
    }
    else if (num_pols == 4)
    {
        corr_type(1) = Stokes::XY;
        corr_type(2) = Stokes::YX;
        corr_type(3) = Stokes::YY;
    }

    // Set up the correlation product, based on number of polarisations.
    Matrix<Int> corr_product(2, num_pols);
    for (unsigned int i = 0; i < num_pols; ++i)
    {
        corr_product(0, i) = Stokes::receptor1(Stokes::type(corr_type(i)));
        corr_product(1, i) = Stokes::receptor2(Stokes::type(corr_type(i)));
    }

    // Create a new row, and fill the columns.
    unsigned int row = p->ms->polarization().nrow();
    p->ms->polarization().addRow();
    p->msc->polarization().corrType().put(row, corr_type);
    p->msc->polarization().corrProduct().put(row, corr_product);
    p->msc->polarization().numCorr().put(row, num_pols);
#endif
}
