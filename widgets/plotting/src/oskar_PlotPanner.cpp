#include "widgets/plotting/oskar_PlotPanner.h"

#include <QtGui/QKeyEvent>

namespace oskar {

PlotPanner::PlotPanner(QwtPlotCanvas * canvas)
: QwtPlotPanner(canvas)
{
    setMouseButton(Qt::MidButton);
}

void PlotPanner::widgetKeyPressEvent(QKeyEvent * e)
{
    int key = e->key();
    int step = 5;
    if (e->modifiers() == Qt::ShiftModifier)
    {
        if (key == Qt::Key_Right)
            moveCanvas(step, 0);
        else if (key == Qt::Key_Left)
            moveCanvas(-step, 0);
        else if (key == Qt::Key_Up)
            moveCanvas(0, -step);
        else if (key == Qt::Key_Down)
            moveCanvas(0, step);
    }
    else if (e->modifiers() == Qt::NoModifier)
    {
        if (key == Qt::Key_A)
            moveCanvas(-step, 0);
        else if (key == Qt::Key_D)
            moveCanvas(step, 0);
        else if (key == Qt::Key_W)
            moveCanvas(0, -step);
        else if (key == Qt::Key_S)
            moveCanvas(0, step);
    }
}


} // namespace oskar
