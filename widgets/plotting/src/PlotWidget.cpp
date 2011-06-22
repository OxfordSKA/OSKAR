#include "widgets/plotting/PlotWidget.h"
#include "widgets/plotting/PlotPicker.h"
#include "widgets/plotting/PlotMagnifier.h"
#include "widgets/plotting/PlotZoomer.h"
#include "widgets/plotting/PlotPanner.h"

#include <qwt_symbol.h>
#include <qwt_scale_widget.h>
#include <qwt_plot_layout.h>
#include <qwt_plot_grid.h>
#include <qwt_plot_printfilter.h>
#include <qwt_scale_map.h>

#include <QtGui/QMenu>
#include <QtCore/QString>
#include <QtGui/QKeyEvent>
#include <QtGui/QFileDialog>
#include <QtGui/QInputDialog>
#include <QtGui/QPrinter>
#include <QtGui/QMessageBox>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
using namespace std;

namespace oskar {

/**
 * @details
 */
PlotWidget::PlotWidget(QWidget* parent)
: QwtPlot(parent), _adjustingColours(false), _oldx(0), _oldy(0), _picker(0),
  _panner(0), _zoomer(0), _magnifier(0), _scale(0), _grid(0)
{
    // Setup the colour scale.
    _scale = new QwtScaleWidget;
    _scale = axisWidget(QwtPlot::yRight);

    // Setup the colour map.
    _colourMap.set(ColourMap::RAINBOW);

    // Setup the plot grid.
    _setGrid();

    // Setup the plot panner.
    _setPanner();

    // Setup the plot zoomer.
    _setZoomer();

//    // Set the curve style (in this case no line, just black crosses).
//    setCurveStyle();

    // Setup the plot Picker.
    _picker = new PlotPicker(canvas());

    // Connect signals to slots.
    _connectSignalsAndSlots();

    // Clear and initialise the widget.
    clear();

    if (!isVisible()) setVisible(true);

    // what does this do? - its from the spectrogram example...
//    plotLayout()->setAlignCanvasToScales(true);

    // Settings for stand alone mode (i.e not embedded in a parent widget).
    if (parent == 0)
    {
        setMinimumSize(500, 500);
        showGrid(false);
        setCanvasBackground(QColor(Qt::white));
    }
}


/**
 * @details
 */
PlotWidget::~PlotWidget()
{
    if (_picker) { delete _picker; _picker = 0; }
    if (_panner) { delete _panner; _panner = 0; }
    if (_grid) { delete _grid; _grid = 0; }
    if (_zoomer) { delete _zoomer; _zoomer = 0; }
    if (_magnifier) { delete _magnifier; _magnifier = 0; }
    if (_scale) { delete _scale; _scale = 0; }
    for (unsigned i = 0; i < _curve.size(); ++i)
    {
        delete _curve[i];
    }
}


/**
 * @details
 */
void PlotWidget::slotSelectedPoint(const QwtDoublePoint& /* point */)
{
}


/**
 * @details
 */
void PlotWidget::slotSelectedRect(const QwtDoubleRect& rectangle)
{
    qreal xTopLeft, yTopLeft, xBottomRight, yBottomRight;
    rectangle.getCoords(&xTopLeft, &yTopLeft, &xBottomRight, &yBottomRight);
    emit xLeftChanged(xTopLeft);
    emit xRightChanged(xBottomRight);
    emit yTopChanged(yTopLeft);
    emit yBottomChanged(yBottomRight);
}


/**
 * @details
 * Clears the plot widget.
 */
void PlotWidget::clear()
{
    setAxisScale(QwtPlot::yLeft, -1.0, 1.0);
    setAxisScale(QwtPlot::xBottom, -1.0, 1.0);
    enableAxis(QwtPlot::yRight, false);
    plotLayout()->setAlignCanvasToScales(true);
    setTitle("");
    setXLabel("");
    setYLabel("");
    updateZoomBase();

    _image.detach();
    for (unsigned i = 0; i < _curve.size(); ++i)
    {
        _curve[i]->detach();
    }

    showGrid(true);

    replot();
}


/**
 * @details
 * Plots a curve specified by xData and yData arrays;
 *
 * @param[in] nPoints   Number of points in the curve.
 * @param[in] xData     X values.
 * @param[in] yData     Y values.
 * @param[in] reverseX  Reverse the x axis.
 * @param[in] reverseY  Reverse the y axis.
 */
void PlotWidget::plotCurve(const unsigned nPoints, const double * xData,
        const double * yData, const bool reverseX, const bool reverseY,
        bool append)
{
    if (nPoints < 1 || xData == 0 || yData == 0)
    {
        cerr << "PlotWidget::plotCurve(): Input data error." << endl;
        return;
    }

    // Detach any image plot data and disable the colour bar.
    _image.detach();
    enableAxis(QwtPlot::yRight, false);

    // Pass the curve data to the curve object.
    // - use setRawData to not make copies...?

    QwtPlotCurve * c = 0;
    if (_curve.size() == 0 || append == true)
    {
        _curve.push_back(new QwtPlotCurve);
        setCurveStyle(_curve.size()-1);
        c = _curve[_curve.size() - 1];
    }
    else if (append == false)
    {
//        for (unsigned i = 0; i < _curve.size();  ++i)
//            delete _curve[i];
//        _curve.push_back(new QwtPlotCurve);
//        setCurveStyle(0);
        c = _curve[0];
    }
//    QwtPlotCurve * c = _curve[_curve.size() - 1];
    c->setData(xData, yData, nPoints);

    if (append == false)
    {
        // Set the axis range.
        double xMin = (reverseX) ? c->maxXValue() : c->minXValue();
        double xMax = (reverseX) ? c->minXValue() : c->maxXValue();
        double yMin = (reverseY) ? c->maxYValue() : c->minYValue();
        double yMax = (reverseY) ? c->minYValue() : c->maxYValue();
        setAxisScale(QwtPlot::xBottom, xMin, xMax);
        setAxisScale(QwtPlot::yLeft, yMin, yMax);

        // Update the plot zoom base stack.
        updateZoomBase();
    }


    // Attach the curve to the plot widget.
    c->attach(this);
}


void PlotWidget::plotCurve(const unsigned nPoints, const float * y,
        const float * x, const QString & title, bool append)
{
    vector<double> _x(nPoints), _y(nPoints);
    for (unsigned i = 0; i < nPoints; ++i)
    {
        _x[i] = (x == NULL) ? (double)i : (double)x[i];
        _y[i] = (double)y[i];
    }
    plotCurve(nPoints, &_x[0], &_y[0], false, false, append);
    setTitle(title);
    setWindowTitle(title);
}



void PlotWidget::plotCurve(const unsigned nPoints, const float * y,
        const QString & title, bool append)
{
    plotCurve(nPoints, y, NULL, title, append);
}



/// Plot an image
void PlotWidget::plotImage(const float * data, unsigned nX, unsigned nY,
        double xMin, double xMax, double yMin, double yMax, bool reverseX,
        bool reverseY)
{
    if (!data || nX < 1 || nY < 1)
    {
        cerr << "PlotWidget::plotImage(): Data Error." << endl;
        return;
    }

    // Detach all curves plot items from the plot widget.
    for (unsigned i = 0; i < _curve.size(); ++i)
        _curve[i]->detach();

    // Set the range and data array of the image plot.
    _image.setImageArray(data, nX, nY, xMin, xMax, yMin, yMax);

    // Set the x axis plot range.
    if (reverseX)
        setAxisScale(QwtPlot::xBottom, xMax, xMin);
    else
        setAxisScale(QwtPlot::xBottom, xMin, xMax);

    // Set the x axis plot range.
    if (reverseY)
        setAxisScale(QwtPlot::yLeft, yMax, yMin);
    else
        setAxisScale(QwtPlot::yLeft, yMin, yMax);

    // Update the plot zoom base stack.
    updateZoomBase();

    // Attach the image plot to the widget.
    _image.attach(this);

    // Enable the colour scale.
    double zMin = _image.ampRange().minValue();
    double zMax = _image.ampRange().maxValue();

    setAxisScale(QwtPlot::yRight, zMin, zMax);
    enableColourScale(true);
}


void PlotWidget::plotImage(const unsigned size, float const * data,
        const QString & title)
{
    const unsigned nX = size;
    const unsigned nY = size;
    const double xmin = 0.0f;
    const double xmax = (double)size;
    const double ymin = 0.0f;
    const double ymax = (double)size;
    plotImage(data, nX, nY, xmin, xmax, ymin, ymax);
    setTitle(title);
    setWindowTitle(title);
}


void PlotWidget::plotImage(const unsigned size, const Complex * data,
                const c_type type, const QString & title)
{
    const unsigned nX = size;
    const unsigned nY = size;
    const double xmin = 0.0f;
    const double xmax = (double)size;
    const double ymin = 0.0f;
    const double ymax = (double)size;
    vector<float> image(size * size);
    for (unsigned i = 0; i < size * size; ++i)
    {
        switch (type)
        {
            case RE:
                image[i] = data[i].real(); break;
            case IM:
                image[i] = data[i].imag(); break;
            case ABS:
                image[i] = abs(data[i]); break;
            case PHASE:
                image[i] = arg(data[i]); break;
            case SQRT:
                image[i] = abs(sqrt(data[i])); break;
            default:
                cerr << "Unknown c_type!" << endl;
                return;
        };
    }
    plotImage(&image[0], nX, nY, xmin, xmax, ymin, ymax);
    setTitle(title);
    setWindowTitle(title);
}



/**
 * @details
 */
void PlotWidget::savePNG(QString fileName, unsigned sizeX, unsigned sizeY)
{
    if (fileName.isEmpty())
    {
        fileName = QFileDialog::getSaveFileName(this,
                "Save plot widget as PNG: file name", QString(), "(*.png)");
    }

    if (fileName.isEmpty()) return;

    QFileInfo fileInfo(fileName);
    if (fileInfo.suffix() != "png")
        fileName += ".png";

    QPixmap pixmap(sizeX, sizeY);
    pixmap.fill();
    print(pixmap);
    int quality = -1; // -1 = default, [0..100] otherwise.
    if (!pixmap.save(fileName, "PNG", quality))
        throw QString("PlotWidget::exportPNG(): Error saving PNG");
}


/**
 *
 * @param fileName
 */
void PlotWidget::savePDF(QString fileName, unsigned sizeX, unsigned sizeY)
{
    if (fileName.isEmpty())
    {
        fileName = QFileDialog::getSaveFileName(this,
                "Save plot widget as PDF: file name", QString(), "(*.pdf)");
    }

    if (fileName.isEmpty()) return;

    QFileInfo fileInfo(fileName);
    if (fileInfo.suffix() != "pdf")
        fileName += ".pdf";

    QPrinter printer(QPrinter::ScreenResolution);
    printer.setDocName(fileName);
    printer.setOutputFileName(fileName);
    printer.setCreator("oskar");
    printer.setColorMode(QPrinter::Color);
    printer.setOutputFormat(QPrinter::PdfFormat);
    printer.setPaperSize(QSizeF(sizeX, sizeY), QPrinter::Point);
    print(printer);
}




/**
 * @details
 */
void PlotWidget::rectangleSelect(bool enable)
{
    if (!_picker) return;
    if (enable) _picker->setSelectionType(PlotPicker::RECT_SELECT);
    else _picker->setSelectionType(PlotPicker::POINT_SELECT);
}


/**
 * @details
 */
void PlotWidget::setCurveStyle(const unsigned iCurve)
{
    QwtSymbol _symbol;
    Qt::GlobalColor col = Qt::black;
    Qt::GlobalColor brushCol = Qt::transparent;
    QwtSymbol::Style sym = QwtSymbol::Cross;
    const int s1 = 5;
    const int s2 = 7;
    QSize size(s1, s1);
    const int iStyle = iCurve%7;
    //cout << "iCurve = " << iCurve << " -> iStyle = " << iStyle << endl;
    switch (iStyle)
    {
        case 0:
            col = Qt::black;
            sym = QwtSymbol::Cross;
            break;
        case 1:
            col = Qt::red;
            sym = QwtSymbol::Ellipse;
            size = QSize(s2, s2);
            break;
        case 2:
            col = Qt::blue;
            sym = QwtSymbol::Diamond;
            size = QSize(s2, s2);
            break;
        case 3:
            col = Qt::green;
            sym = QwtSymbol::Triangle;
            brushCol = Qt::green;
            size = QSize(s2, s2);
            break;
        case 4:
            col = Qt::magenta;
            sym = QwtSymbol::Rect;
            size = QSize(s2, s2);
            break;
        case 5:
            col = Qt::black;
            sym = QwtSymbol::Hexagon;
            size = QSize(s2, s2);
            break;
        case 6:
            col = Qt::cyan;
            sym = QwtSymbol::Star1;
            size = QSize(s2, s2);
            break;
        default:
        {}
    };

    _symbol.setStyle(sym);
    _symbol.setPen(QPen(QColor(col)));
    _symbol.setBrush(QBrush(brushCol));
    _symbol.setSize(size);
    _curve[iCurve]->setSymbol(_symbol);
    _curve[iCurve]->setStyle(QwtPlotCurve::NoCurve);
}


/**
 * @details
 */
void PlotWidget::updateZoomBase()
{
    if (!_zoomer) return;
    _zoomer->setZoomBase(true);
}


/**
 * @details
 */
void PlotWidget::menuSetTitle()
{
    bool ok = false;
    QString text = QInputDialog::getText(this, "Set Plot Title", "Title: ",
            QLineEdit::Normal, title().text(), &ok);

    if (ok && text.isEmpty()) setTitle("");
    if (ok && !text.isEmpty()) setTitle(text);
}


/**
 * @details
 */
void PlotWidget::setPlotTitle(const QString& text)
{
    QFont font;
    font.setPointSize(11);
    QwtText label(text);
    label.setFont(font);
    setTitle(label);
    emit titleChanged(text);
}


/**
 * @details
 */
void PlotWidget::setXLabel(const QString& text)
{
    QFont font;
    font.setPointSize(9);
    QwtText label(text);
    label.setFont(font);
    setAxisTitle(QwtPlot::xBottom, label);
    emit xLabelChanged(text);
}


/**
 * @details
 */
void PlotWidget::menuSetXLabel()
{
    bool ok = false;
    QString t = QInputDialog::getText(this, "Set X Label", "Label: ",
            QLineEdit::Normal, axisTitle(QwtPlot::xBottom).text(), &ok);
    if (ok && t.isEmpty()) setAxisTitle(QwtPlot::xBottom, "");
    if (ok && !t.isEmpty()) setXLabel(t);
}

/**
 * @details
 */
void PlotWidget::menuSetYLabel()
{
    bool ok = false;
    QString t = QInputDialog::getText(this, "Set Y Label", "Label: ",
            QLineEdit::Normal, axisTitle(QwtPlot::yLeft).text(), &ok);

    if (ok && t.isEmpty()) setAxisTitle(QwtPlot::yLeft, "");
    else if (ok && !t.isEmpty()) setYLabel(t);
}


/**
 * @details
 */
void PlotWidget::setYLabel(const QString& text)
{
    QFont font;
    font.setPointSize(9);
    QwtText label(text);
    label.setFont(font);
    setAxisTitle(QwtPlot::yLeft, label);
    emit yLabelChanged(text);
}


/**
 *
 * @param text
 */
void PlotWidget::setScaleLabel(const QString& text)
{
    QFont scaleFont;
    scaleFont.setPointSize(9);
    QwtText scaleLabel(text);
    scaleLabel.setFont(scaleFont);
    _scale->setTitle(scaleLabel);
}


/**
 * @details
 */
void PlotWidget::setAxesVisible(int state)
{
    bool visible = (state == Qt::Checked) ? true : false;
    enableAxis(QwtPlot::xBottom, visible);
    enableAxis(QwtPlot::yLeft, visible);
}


/**
 * @details
 *
 * @param on
 */
void PlotWidget::showGrid(bool on)
{
    if (!_grid) return;
    _grid->setVisible(on);
    replot();
    emit gridEnabled(on);
}




/**
 * @details
 */
void PlotWidget::showGridMinorTicks(bool on)
{
    if (!_grid) return;
    _grid->enableXMin(on);
    _grid->enableYMin(on);
    replot();
    emit gridMinorTicks(on);
}

void PlotWidget::toggleGrid()
{
    // If the grid isn't setup return;
    if (!_grid) return;

    // Toggle the grid visibility.
    _grid->setVisible(!_grid->isVisible());

    // Re-draw.
    replot();

    // Emit the grid visibility signal.
    emit gridEnabled(_grid->isVisible());
}


void PlotWidget::toggleGridMinorTicks()
{
    // If the grid isn't setup return;
    if (!_grid) return;

    // Don't do anything if the grid is disabled.
    if (!_grid->isVisible()) return;

    // Toggle minor grid ticks.
    _grid->enableXMin(!_grid->xMinEnabled());
    _grid->enableYMin(!_grid->yMinEnabled());

    replot();

    emit gridMinorTicks(_grid->xMinEnabled());
}



/**
 * @details
 */
void PlotWidget::enableColourScale(bool on)
{
    QList<QwtPlotItem*> items = itemList();
    if (!items.contains(&_image)) return;

    if (!_scale) return;
    if (on) {
        _image.setColorMap(_colourMap.map());
        _scale->setColorMap(_image.ampRange(), _image.colorMap());
        _scale->setColorBarEnabled(true);
        enableAxis(QwtPlot::yRight, true);
    }
    else {
        _scale->setColorBarEnabled(false);
        enableAxis(QwtPlot::yRight, false);
    }
}


void PlotWidget::togglePanner(bool on)
{
    if (!_panner) return;
    _panner->setEnabled(on);
}


void PlotWidget::toggleZoomer(bool on)
{
    if (!_zoomer) return;
    _zoomer->setEnabled(on);
}


void PlotWidget::togglePicker(bool on)
{
    if (!_picker) return;
    _picker->setEnabled(on);
}

/**
 * @details
 * Process the mouse press event.
 */
void PlotWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::RightButton && event->modifiers() == Qt::ShiftModifier)
    {
        _oldx = event->x();
        _oldy = event->y();
        _adjustingColours = true;
    }
}


/**
 * @details
 * Process the mouse move event. If adjusting colours (set by a mouse
 * press event) the colour map is updated.
 *
 * @param[in] event Qt Mouse event object pointer.
 */
void PlotWidget::mouseMoveEvent(QMouseEvent* event)
{
    float sensitivity = 800.0f;

    if (_adjustingColours)
    {
        int x = event->x();
        int y = event->y();
        float c = _colourMap.contrast() + float(x - _oldx) / sensitivity;
        float b = _colourMap.brightness() + float(y - _oldy) / sensitivity;
        _oldx = x;
        _oldy = y;
        _colourMap.update(b, c);

        _image.setColorMap(_colourMap.map());
        enableColourScale();
        replot();
    }
}


/**
 * @details
 * Process the mouse release event.
 */
void PlotWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::RightButton && _adjustingColours)
        _adjustingColours = false;
}


/**
 *
 * @param
 */
void PlotWidget::keyPressEvent(QKeyEvent* event)
{
    // Keys with shift modifier.
    if (event->modifiers() == Qt::ShiftModifier)
    {
        if (event->key() == Qt::Key_G)
            toggleGridMinorTicks();

        if (event->key() == Qt::Key_P)
            savePDF();
    }

    else if (event->modifiers() == Qt::AltModifier)
    {
        if (event->key() == Qt::Key_C)
            clear();
    }

    else if (event->modifiers() == Qt::ControlModifier)
    {
        if (event->key() == Qt::Key_C)
        {
            if (parentWidget() == 0)
                close();
        }
        else if (event->key() == Qt::Key_X)
            close();

    }
    // Unmodified keys.
    else if (event->modifiers() == Qt::NoModifier)
    {
        if (event->key() == Qt::Key_G)
            toggleGrid();

        else if (event->key() == Qt::Key_F5)
        {
            _colourMap.update();
            _oldx = 0;
            _oldy = 0;
            _zoomer->zoom(_zoomer->zoomBase());
            //_magnifier->rescale(1.0);
            replot();
        }

        else if (event->key() == Qt::Key_T)
            menuSetTitle();

        else if (event->key() == Qt::Key_X)
            menuSetXLabel();

        else if (event->key() == Qt::Key_Y)
            menuSetYLabel();

        else if (event->key() == Qt::Key_P)
            savePNG();

        else if (event->key() == Qt::Key_Escape)
        {
            if (parentWidget() == 0)
                close();
        }

        else if (event->key() == Qt::Key_F1)
        {
            QString h;
            h += "Keys:\n";
            h += "========================\n";
            h += "t\t Set the plot title.\n";
            h += "x\t Set the plot x label.\n";
            h += "y\t Set the plot y label.\n";
            h += "\n";
            h +=  "g\t Toggle grid.\n";
            h += "[shift] + g\t Toggle grid minor ticks.\n";
            h += "\n";
            h += "p\t Save the plot as a PNG.\n";
            h += "[shift] + p\t Save the plot as a PDF.\n";
            h += "\n";
            h += "F5\t Refresh the plot.\n";
            h += "\n";
            h += ".\tZoom in one step.\n";
            h += "/\tZoom out one step.\n";
            h += "\n";
            h += "w\tPan up.\n";
            h += "a\tPan left.\n";
            h += "s\tPan down.\n";
            h += "d\tPan right.\n";
            h += "\n";
            h += "q\tReset the plot (zoom/pan etc.).\n";
            h += "\n";
            h += "Esc\tClose the widget (safer - only works if no parent).\n";
            h += "[ctrl] + c\tClose the plot (same as above).\n";
            h += "\n";
            h += "[alt] + c\tClear the plot (use with care!).\n";
            h += "[ctrl] + x\tClose the widget (use with care!).\n";
            h += "\n";
            h += "Mouse:\n";
            h += "========================\n";
            h += "left-click\t\tZoom rectangle.\n";
            h += "right-click\t\tZoom out one step.\n";
            h += "[ctrl] + right-click\tZoom out fully.\n";
            h += "[shift] + right-click\tDisplay cross-hair position.\n";
            h += "[shift] + right-drag\tAdjust colour map.\n";
            h += "middle-drag\t\tPan plot.\n";
            h += "[shift] + wheel\tMagnify plot.\n";

            QMessageBox::information(this, "Key-bindings help", h);
        }
    }
}


/**
 * @details
 */
void PlotWidget::_setGrid()
{
    // Already initialised? return.
    if (_grid) return;

    _grid = new QwtPlotGrid();
    _grid->attach(this);

    _grid->setMinPen(QPen(QBrush(QColor(Qt::lightGray)), qreal(0.0), Qt::DotLine));
    _grid->setMajPen(QPen(QBrush(QColor(Qt::darkGray)),  qreal(0.0), Qt::DotLine));

    _grid->enableXMin(false);
    _grid->enableYMin(false);
}


/**
 * @details
 */
void PlotWidget::_setPanner()
{
    // Already initialised? return.
    if (_panner) return;
    _panner = new PlotPanner(canvas());
    _panner->setAxisEnabled(QwtPlot::yRight, false);
}


/**
 * @details
 */
void PlotWidget::_setZoomer()
{
    if (_zoomer == 0)
    {
        _zoomer = new PlotZoomer(canvas());
    }

    if (_magnifier == 0)
    {
        _magnifier = new PlotMagnifier(canvas());
        _magnifier->setAxisEnabled(QwtPlot::yRight, false);
    }
}


/**
 * @details
 * Sets signal and slot connections.
 */
void PlotWidget::_connectSignalsAndSlots()
{
    // Signals emitted by the plot picker.
    // =========================================================================
    connect(_picker, SIGNAL(selected(const QwtDoublePoint &)),
            this, SLOT(slotSelectedPoint(const QwtDoublePoint &)));
    connect(_picker, SIGNAL(selected(const QwtDoubleRect &)),
            this, SLOT(slotSelectedRect(const QwtDoubleRect &)));

    // Signals emitted by the colour map.
    // =========================================================================
    connect(&_colourMap, SIGNAL(mapUpdated()),
            this, SLOT(enableColourScale()));
}

} // namespace oskar
