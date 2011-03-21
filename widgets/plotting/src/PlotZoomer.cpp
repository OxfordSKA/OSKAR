#include "widgets/plotting/PlotZoomer.h"

#include <QtGui/QPen>
#include <QtGui/QBrush>
#include <QtGui/QColor>

namespace oskar {

PlotZoomer::PlotZoomer(QwtPlotCanvas * canvas)
: QwtPlotZoomer(canvas)
{
    setTrackerMode(QwtPicker::ActiveOnly);
    setRubberBandPen(QColor(Qt::green));
    setSelectionFlags(QwtPicker::CornerToCorner | QwtPicker::DragSelection);

    // Select a zoom rectangle.
    setMousePattern(QwtEventPattern::MouseSelect1, Qt::LeftButton);

    // Unzoom to initial size.
    setMousePattern(QwtEventPattern::MouseSelect2, Qt::RightButton, Qt::ControlModifier);
    setKeyPattern(QwtEventPattern::KeyHome, Qt::Key_Q);

    // Zoom out one step at a time.
    setMousePattern(QwtEventPattern::MouseSelect3, Qt::RightButton);
}


QwtText PlotZoomer::trackerText(const QwtDoublePoint &pos) const
{
    QColor bg(Qt::white);
    bg.setAlpha(200);

    QwtText text = QwtPlotZoomer::trackerText(pos);
    text.setBackgroundBrush(QBrush(bg));

    return text;
}

} // namespace oskar
