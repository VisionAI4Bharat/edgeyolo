/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#include <QApplication>
#include <QCommandLineParser>
#include <opencv2/core.hpp>
#include "ui/mainwindow.h"
#include "app_config.h"
#include "debug_log.h"

int main(int argc, char *argv[]) {
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<std::vector<inference::Detection>>("std::vector<inference::Detection>");

    QApplication app(argc, argv);
    app.setApplicationName("deepSightAI");
    app.setOrganizationName("deepSightAI");

    QCommandLineParser parser;
    parser.setApplicationDescription("deepSightAI inference GUI");
    parser.addHelpOption();
    parser.addOption({{"c", "config"}, "Path to config YAML file.", "config.yaml"});
    parser.process(app);

    MainWindow window;
    window.show();

    if (parser.isSet("config"))
        window.dsai_loadFromConfigFile(parser.value("config"));
    else
        window.dsai_loadFromConfigFile(QString::fromStdString(AppConfig::dsai_defaultPath()));

    return app.exec();
}
