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
