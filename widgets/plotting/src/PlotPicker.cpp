#include "widgets/plotting/PlotPicker.h"
#include <QtCore/QString>
#include <iostream>
#include <cstdio>

#include <qwt_plot.h>
#include <qwt_plot_spectrogram.h>
#include "widgets/plotting/ImagePlotData.h"
#include "widgets/plotting/ImagePlot.h"

using namespace std;

namespace oskar {


/**
 * @details
 * Constructor for Picker object. Sets various defaults.
 */
PlotPicker::PlotPicker(QwtPlotCanvas * canvas)
: QwtPlotPicker(canvas)
{
    setSelectionType(POINT_SELECT);
    setMousePattern(QwtEventPattern::MouseSelect1, Qt::LeftButton, Qt::ShiftModifier);
    setRubberBandPen(QPen(QColor(Qt::green)));
    setTrackerPen(QPen(QColor(Qt::darkBlue)));
    setTrackerMode(QwtPicker::ActiveOnly);
}


/**
 * @details
 */
void PlotPicker::setSelectionType(unsigned type)
{
    switch (type)
    {
        case RECT_SELECT:
        {
            setSelectionFlags(QwtPicker::RectSelection | QwtPicker::DragSelection);
            setRubberBand(QwtPicker::RectRubberBand);
            break;
        }

        case POINT_SELECT:
        {
            setSelectionFlags(QwtPicker::PointSelection | QwtPicker::DragSelection);
            setRubberBand(QwtPicker::CrossRubberBand);
            break;
        }
        default:
            throw QString("PlotPicker: Unknown selection type.");
    }
}

/**
* @details
* Function to translate a pixel position into a string
*/
QwtText PlotPicker::trackerText(const QwtDoublePoint& pos) const
{
    const QwtPlot * p = this->plot();
    QwtPlotItemList pl = p->itemList();
    const float x = pos.x();
    const float y = pos.y();
//    printf("number of items = %d\n", pl.size());
    float value;
    QwtText text;
    for (unsigned i = 0; i < (unsigned)pl.size(); ++i)
    {
        int type = pl[i]->rtti();
        if (type == QwtPlotItem::Rtti_PlotSpectrogram)
        {
            ImagePlot * s = (ImagePlot*)pl[i];
            const ImagePlotData * sd = s->getData();
            value = sd->value((double)x, (double)y);
            int col = sd->column((double)x);
            int row = sd->row((double)y);
            text = "(" + QString::number(col) + ", " +
                    QString::number(row) + "): " +
                    QString::number(value, 'e', 8);
//            printf("spectrogram(%s) %s\n", s->plot()->title().text().toLatin1().data(),
//                    text.text().toLatin1().data());
        }
        else if (type == QwtPlotItem::Rtti_PlotCurve)
        {
            text = "(" + QString::number(x, 'f', 2) + ", " + QString::number(y, 'f', 2) + ")";
//            printf("curve: %s\n", text.text().toLatin1().data());
        }
        else
        {
//            text = "";
        }
    }

    QColor colour(Qt::white);
    colour.setAlpha(230);

    text.setBackgroundBrush(QBrush(colour));
    return text;
}


} // namespace oskar
