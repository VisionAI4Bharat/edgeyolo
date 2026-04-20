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

    QStringList getClassLabels() const;

private slots:
    void addLabel();
    void deleteLabel();
    void moveUp();
    void moveDown();
    void readFromYaml();
    void updateButtonStates();

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
