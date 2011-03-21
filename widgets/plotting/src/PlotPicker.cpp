#include "widgets/plotting/PlotPicker.h"
#include <QtCore/QString>
#include <iostream>

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


///**
//* @details
//* Function to translate a pixel position into a string
//*/
//QwtText PlotPicker::trackerText(const QPoint& pos) const
//{
//    QColor colour(Qt::white);
//    colour.setAlpha(230);
//    const float x = pos.x();
//    const float y = pos.y();
//    QwtText text = QString::number(x,'f',4) + ", " + QString::number(y,'f',4);
//    text.setBackgroundBrush(QBrush(colour));
//    return text;
//}


/**
* @details
* Function to translate a pixel position into a string
*/
QwtText PlotPicker::trackerText(const QwtDoublePoint& pos) const
{
    QColor colour(Qt::white);
    colour.setAlpha(230);
    const float x = pos.x();
    const float y = pos.y();
    QwtText text = QString::number(x,'f',4) + ", " + QString::number(y,'f',4);
    text.setBackgroundBrush(QBrush(colour));
    return text;
}


} // namespace oskar
