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

#include "classlabelsdialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFrame>
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <fstream>
#include <yaml-cpp/yaml.h>

ClassLabelsDialog::ClassLabelsDialog(const QStringList& labels,
                                     const QString&     yamlPath,
                                     QWidget*           parent)
    : QDialog(parent), yamlPath_(yamlPath)
{
    setWindowTitle("Edit Class Labels");
    setModal(true);
    setMinimumSize(420, 380);

    // ── list + side buttons ───────────────────────────────────────────────────
    list_ = new QListWidget(this);
    list_->setDragDropMode(QAbstractItemView::InternalMove);
    for (const QString& lbl : labels) {
        auto* item = new QListWidgetItem(lbl, list_);
        item->setFlags(item->flags() | Qt::ItemIsEditable);
    }

    addBtn_      = new QPushButton("Add",       this);
    deleteBtn_   = new QPushButton("Delete",    this);
    moveUpBtn_   = new QPushButton("Move Up",   this);
    moveDownBtn_ = new QPushButton("Move Down", this);

    auto* sep = new QFrame(this);
    sep->setFrameShape(QFrame::HLine);
    sep->setFrameShadow(QFrame::Sunken);

    QVBoxLayout* btnCol = new QVBoxLayout();
    btnCol->addWidget(addBtn_);
    btnCol->addWidget(deleteBtn_);
    btnCol->addWidget(sep);
    btnCol->addWidget(moveUpBtn_);
    btnCol->addWidget(moveDownBtn_);
    btnCol->addStretch();

    QHBoxLayout* listArea = new QHBoxLayout();
    listArea->addWidget(list_, 1);
    listArea->addLayout(btnCol);

    QGroupBox* group = new QGroupBox("Class Labels", this);
    group->setLayout(listArea);

    // ── bottom button row ─────────────────────────────────────────────────────
    readYamlBtn_          = new QPushButton("Read from YAML…", this);
    auto* applyBtn        = new QPushButton("Apply",           this);
    auto* cancelBtn       = new QPushButton("Cancel",          this);
    applyBtn->setDefault(true);

    QHBoxLayout* bottom = new QHBoxLayout();
    bottom->addWidget(readYamlBtn_);
    bottom->addStretch();
    bottom->addWidget(applyBtn);
    bottom->addWidget(cancelBtn);

    // ── main layout ───────────────────────────────────────────────────────────
    QVBoxLayout* main = new QVBoxLayout(this);
    main->addWidget(group, 1);
    main->addLayout(bottom);

    connect(addBtn_,      &QPushButton::clicked, this, &ClassLabelsDialog::dsai_addLabel);
    connect(deleteBtn_,   &QPushButton::clicked, this, &ClassLabelsDialog::dsai_deleteLabel);
    connect(moveUpBtn_,   &QPushButton::clicked, this, &ClassLabelsDialog::dsai_moveUp);
    connect(moveDownBtn_, &QPushButton::clicked, this, &ClassLabelsDialog::dsai_moveDown);
    connect(readYamlBtn_, &QPushButton::clicked, this, &ClassLabelsDialog::dsai_readFromYaml);
    connect(applyBtn,     &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelBtn,    &QPushButton::clicked, this, &QDialog::reject);
    connect(list_, &QListWidget::currentRowChanged, this, &ClassLabelsDialog::dsai_updateButtonStates);

    dsai_updateButtonStates();
}

QStringList ClassLabelsDialog::dsai_getClassLabels() const
{
    QStringList result;
    for (int i = 0; i < list_->count(); ++i) {
        const QString t = list_->item(i)->text().trimmed();
        if (!t.isEmpty())
            result << t;
    }
    return result;
}

void ClassLabelsDialog::dsai_addLabel()
{
    int row = list_->currentRow();
    int insertAt = (row >= 0) ? row + 1 : list_->count();

    auto* item = new QListWidgetItem("New Label");
    item->setFlags(item->flags() | Qt::ItemIsEditable);
    list_->insertItem(insertAt, item);
    list_->setCurrentItem(item);
    list_->editItem(item);
    dsai_updateButtonStates();
}

void ClassLabelsDialog::dsai_deleteLabel()
{
    const int row = list_->currentRow();
    if (row < 0) return;
    delete list_->takeItem(row);
    dsai_updateButtonStates();
}

void ClassLabelsDialog::dsai_moveUp()
{
    const int row = list_->currentRow();
    if (row <= 0) return;
    QListWidgetItem* item = list_->takeItem(row);
    list_->insertItem(row - 1, item);
    list_->setCurrentRow(row - 1);
    dsai_updateButtonStates();
}

void ClassLabelsDialog::dsai_moveDown()
{
    const int row = list_->currentRow();
    if (row < 0 || row >= list_->count() - 1) return;
    QListWidgetItem* item = list_->takeItem(row);
    list_->insertItem(row + 1, item);
    list_->setCurrentRow(row + 1);
    dsai_updateButtonStates();
}

void ClassLabelsDialog::dsai_readFromYaml()
{
    QString path = yamlPath_;
    if (path.isEmpty() || !QFile::exists(path)) {
        path = QFileDialog::getOpenFileName(
            this, "Select Class Labels YAML",
            path.isEmpty() ? QString() : QFileInfo(path).absolutePath(),
            "YAML Files (*.yaml *.yml);;All Files (*)");
        if (path.isEmpty()) return;
        yamlPath_ = path;
    }

    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(path.toStdString());
    } catch (const YAML::Exception& e) {
        QMessageBox::warning(this, "YAML Error",
            QString("Failed to parse YAML:\n%1").arg(e.what()));
        return;
    }

    // Try class_labels first, fall back to names for legacy files
    YAML::Node node;
    if (cfg["class_labels"])
        node = cfg["class_labels"];
    else if (cfg["names"])
        node = cfg["names"];
    else {
        QMessageBox::warning(this, "YAML Error",
            "YAML has neither 'class_labels' nor 'names' key.");
        return;
    }

    list_->clear();
    for (const auto& entry : node) {
        auto* item = new QListWidgetItem(QString::fromStdString(entry.as<std::string>()));
        item->setFlags(item->flags() | Qt::ItemIsEditable);
        list_->addItem(item);
    }
    dsai_updateButtonStates();
}

void ClassLabelsDialog::dsai_updateButtonStates()
{
    const int row   = list_->currentRow();
    const int count = list_->count();
    deleteBtn_->setEnabled(row >= 0);
    moveUpBtn_->setEnabled(row > 0);
    moveDownBtn_->setEnabled(row >= 0 && row < count - 1);
}
