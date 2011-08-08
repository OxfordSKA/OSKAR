#include "widgets/plotting/oskar_PlotMagnifier.h"

#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>

namespace oskar {

PlotMagnifier::PlotMagnifier(QwtPlotCanvas * canvas)
: QwtPlotMagnifier(canvas)
{
    setZoomInKey(Qt::Key_Period, Qt::NoModifier);
    setZoomOutKey(Qt::Key_Slash, Qt::NoModifier);
}


void PlotMagnifier::widgetMousePressEvent(QMouseEvent * /*e*/)
{
    // This disables the default mouse right click and drag zoom.
}


void PlotMagnifier::widgetMouseReleaseEvent(QMouseEvent * /*e*/)
{
    // This disables the default mouse right click and drag zoom.
}


void PlotMagnifier::widgetWheelEvent(QWheelEvent * e)
{
    // This disables the default mouse wheel zoom by overriding the virtual
    // function to do nothing.
    if (e->modifiers() == Qt::ShiftModifier)
    {
        rescale((e->delta() > 0.0) ? 1.1 : 0.9);
    }
}

} // namespace oskar
