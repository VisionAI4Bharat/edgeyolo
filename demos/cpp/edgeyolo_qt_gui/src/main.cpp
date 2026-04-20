#include <QApplication>
#include <QCommandLineParser>
#include "ui/mainwindow.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("deepSightAI");
    app.setOrganizationName("deepSightAI");

    QCommandLineParser parser;
    parser.setApplicationDescription("deepSightAI inference GUI");
    parser.addHelpOption();
    parser.addOption(QCommandLineOption(
        {"c", "config"},
        "Load configuration from YAML file and start immediately (skips the config dialog).",
        "config.yaml"));
    parser.process(app);

    MainWindow window;
    window.show();

    if (parser.isSet("config"))
        window.loadFromConfigFile(parser.value("config"));

    return app.exec();
}
