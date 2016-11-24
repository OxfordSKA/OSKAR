/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#include "ms/oskar_measurement_set.h"
#include "ms/private_ms.h"

#include <tables/Tables.h>
#include <casa/Arrays/Vector.h>

using namespace casa;

// Method based on CASA VisModelData.cc.
static bool is_otf_model_defined(const String& key, const MeasurementSet& ms)
{
    // Try the Source table.
    if (Table::isReadable(ms.sourceTableName()) && ms.source().nrow() > 0 &&
            ms.source().keywordSet().isDefined(key))
        return true;

    // Try the Main table.
    if (ms.keywordSet().isDefined(key))
        return true;
    return false;
}

// Method based on CASA VisModelData.cc.
static bool is_otf_model_defined(const int field,
        const MeasurementSet& ms, String& key, int& source_row)
{
    source_row = -1;
    String mod_key = String("definedmodel_field_") + String::toString(field);
    key = "";
    if (Table::isReadable(ms.sourceTableName()) && ms.source().nrow() > 0)
    {
        if (ms.source().keywordSet().isDefined(mod_key))
        {
            key = ms.source().keywordSet().asString(mod_key);
            if (ms.source().keywordSet().isDefined(key))
                source_row = ms.source().keywordSet().asInt(key);
        }
    }
    else
    {
        if (ms.keywordSet().isDefined(mod_key))
            key = ms.keywordSet().asString(mod_key);
    }
    if (key != "")
        return is_otf_model_defined(key, ms);
    return false;
}

// Method based on CASA VisModelData.cc.
static void remove_field_by_key(MeasurementSet& ms, const String& key)
{
    if (Table::isReadable(ms.sourceTableName()) && ms.source().nrow() > 0 &&
            ms.source().keywordSet().isDefined(key))
    {
        // Replace the source model with an empty record.
        int row = ms.source().keywordSet().asInt(key);
        TableRecord record;
        MSSourceColumns srcCol(ms.source());
        srcCol.sourceModel().put(row, record);
        ms.source().rwKeywordSet().removeField(key);
    }

    // Remove from Main table.
    if (ms.rwKeywordSet().isDefined(key))
        ms.rwKeywordSet().removeField(key);
}

// Method based on CASA VisModelData.cc.
static void remove_otf_model(MeasurementSet& ms)
{
    if (!ms.isWritable())
        return;
    Vector<String> parts(ms.getPartNames(True));
    if (parts.nelements() > 1)
    {
        for (unsigned int k = 0; k < parts.nelements(); ++k)
        {
            MeasurementSet subms(parts[k], ms.lockOptions(), Table::Update);
            remove_otf_model(subms);
        }
        return;
    }

    ROMSColumns msc(ms);
    Vector<Int> fields = msc.fieldId().getColumn();
    int num_fields = GenSort<Int>::sort(fields, Sort::Ascending,
            Sort::HeapSort | Sort::NoDuplicates);
    for (int k = 0; k < num_fields; ++k)
    {
        String key, mod_key;
        int srow;
        if (is_otf_model_defined(fields[k], ms, key, srow))
        {
            mod_key = String("definedmodel_field_") + String::toString(fields[k]);

            // Remove from Source table.
            remove_field_by_key(ms, key);
            if (srow > -1 && ms.source().keywordSet().isDefined(mod_key))
                ms.source().rwKeywordSet().removeField(mod_key);

            // Remove from Main table.
            if (ms.rwKeywordSet().isDefined(mod_key))
                ms.rwKeywordSet().removeField(mod_key);
        }
    }
}


// Method based on CASA VisSetUtil.cc
void oskar_ms_add_scratch_columns(oskar_MeasurementSet* p, int add_model,
        int add_corrected)
{
    if (!p->ms) return;

    // Check if columns need adding.
    add_model = add_model &&
            !(p->ms->tableDesc().isColumn("MODEL_DATA"));
    add_corrected = add_corrected &&
            !(p->ms->tableDesc().isColumn("CORRECTED_DATA"));

    // Return if there's nothing to be done.
    if (!add_model && !add_corrected)
        return;

    // Remove SORTED_TABLE, because old SORTED_TABLE won't see the new columns.
    if (p->ms->keywordSet().isDefined("SORT_COLUMNS"))
        p->ms->rwKeywordSet().removeField("SORT_COLUMNS");
    if (p->ms->keywordSet().isDefined("SORTED_TABLE"))
        p->ms->rwKeywordSet().removeField("SORTED_TABLE");

    // Remove any OTF model data from the MS.
    if (add_model)
        remove_otf_model(*(p->ms));

    // Define a column accessor to the observed data.
    TableColumn* data;
    if (p->ms->tableDesc().isColumn(MS::columnName(MS::FLOAT_DATA)))
        data = new ArrayColumn<Float>(*(p->ms), MS::columnName(MS::FLOAT_DATA));
    else
        data = new ArrayColumn<Complex>(*(p->ms), MS::columnName(MS::DATA));

    // Check if the data column is tiled and, if so, get the tile shape used.
    TableDesc td = p->ms->actualTableDesc();
    const ColumnDesc& column_desc = td[data->columnDesc().name()];
    String dataManType = column_desc.dataManagerType();
    String dataManGroup = column_desc.dataManagerGroup();
    IPosition dataTileShape;
    bool tiled = dataManType.contains("Tiled");
    bool simpleTiling = false;

    if (tiled)
    {
        ROTiledStManAccessor tsm(*(p->ms), dataManGroup);
        unsigned int num_hypercubes = tsm.nhypercubes();

        // Find tile shape.
        int highestProduct = -INT_MAX, highestId = 0;
        for (unsigned int i = 0; i < num_hypercubes; i++)
        {
            int product = tsm.getTileShape(i).product();
            if (product > 0 && (product > highestProduct))
            {
                highestProduct = product;
                highestId = i;
            }
        }
        dataTileShape = tsm.getTileShape(highestId);
        simpleTiling = (dataTileShape.nelements() == 3);
    }

    if (!tiled || !simpleTiling)
    {
        // Untiled, or tiled at a higher than expected dimensionality.
        // Use a canonical tile shape of 1 MB size.
        MSSpWindowColumns msspwcol(p->ms->spectralWindow());
        int max_num_channels = max(msspwcol.numChan().getColumn());
        int tileSize = max_num_channels / 10 + 1;
        int nCorr = data->shape(0)(0);
        dataTileShape = IPosition(3, nCorr,
                tileSize, 131072/nCorr/tileSize + 1);
    }
    delete data;

    if (add_model)
    {
        // Add the MODEL_DATA column.
        TableDesc tdModel;
        String col = MS::columnName(MS::MODEL_DATA);
        tdModel.addColumn(ArrayColumnDesc<Complex>(col, 2));
        td.addColumn(ArrayColumnDesc<Complex>(col, 2));
        MeasurementSet::addColumnToDesc(tdModel,
                MeasurementSet::MODEL_DATA, 2);
        TiledShapeStMan tsm("ModelTiled", dataTileShape);
        p->ms->addColumn(tdModel, tsm);
    }
    if (add_corrected)
    {
        // Add the CORRECTED_DATA column.
        TableDesc tdCorr;
        String col = MS::columnName(MS::CORRECTED_DATA);
        tdCorr.addColumn(ArrayColumnDesc<Complex>(col, 2));
        td.addColumn(ArrayColumnDesc<Complex>(col, 2));
        MeasurementSet::addColumnToDesc(tdCorr,
                MeasurementSet::CORRECTED_DATA, 2);
        TiledShapeStMan tsm("CorrectedTiled", dataTileShape);
        p->ms->addColumn(tdCorr, tsm);
    }
    p->ms->flush();
}

