/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 *
 * This software is dual-licensed:
 * 1. GNU General Public License v3.0 (GPLv3)
 * 2. A proprietary license for commercial use.
 *
 * You may use this software under the terms of the GPLv3 if you are using it
 * for non-commercial purposes. For commercial usage, a separate commercial 
 * license must be obtained from swatah.ai (info@swatah.ai).
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
 * for more details.
 *
 * Trademarks: All trademarks, service marks, and logos are the property of 
 * their respective owners.
 */

#ifndef CLASSLABELSDIALOG_H
#define CLASSLABELSDIALOG_H

#include <QDialog>
#include <QListWidget>
#include <QPushButton>
#include <QStringList>

class ClassLabelsDialog : public QDialog {
    Q_OBJECT
public:
    explicit ClassLabelsDialog(const QStringList& labels,
                               const QString&     yamlPath,
                               QWidget*           parent = nullptr);

    QStringList dsai_getClassLabels() const;

private slots:
    void dsai_addLabel();
    void dsai_deleteLabel();
    void dsai_moveUp();
    void dsai_moveDown();
    void dsai_readFromYaml();
    void dsai_updateButtonStates();

private:
    QListWidget* list_         = nullptr;
    QPushButton* addBtn_       = nullptr;
    QPushButton* deleteBtn_    = nullptr;
    QPushButton* moveUpBtn_    = nullptr;
    QPushButton* moveDownBtn_  = nullptr;
    QPushButton* readYamlBtn_  = nullptr;

    QString yamlPath_;
};

#endif // CLASSLABELSDIALOG_H
