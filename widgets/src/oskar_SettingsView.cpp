#include "widgets/oskar_SettingsView.h"

oskar_SettingsView::oskar_SettingsView(QWidget* parent)
: QTreeView(parent)
{
    connect(this, SIGNAL(expanded(const QModelIndex&)),
            this, SLOT(resizeAfterExpand(const QModelIndex&)));
}

void oskar_SettingsView::resizeAfterExpand(const QModelIndex& /*index*/)
{
    resizeColumnToContents(0);
}
