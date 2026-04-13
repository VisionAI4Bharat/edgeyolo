#include <QApplication>
#include "ui/mainwindow.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("EdgeYOLO Qt GUI");
    app.setOrganizationName("EdgeYOLO");

    MainWindow window;
    window.show();

    return app.exec();
}
