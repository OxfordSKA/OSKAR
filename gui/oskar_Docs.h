/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_DOCS_H_
#define OSKAR_DOCS_H_

#include <QDialog>

namespace oskar {

class Docs : public QDialog
{
    Q_OBJECT

public:
    Docs(QWidget *parent = 0);
};

} /* namespace oskar */

#endif /* include guard */
