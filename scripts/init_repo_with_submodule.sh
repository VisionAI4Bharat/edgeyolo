#!/usr/bin/env bash
# Copyright (C) 2026 swatah.ai. All rights reserved.
#
# This software is dual-licensed:
# 1. GNU General Public License v3.0 (GPLv3)
# 2. A proprietary license for commercial use.
#
# You may use this software under the terms of the GPLv3 if you are using it
# for non-commercial purposes. For commercial usage, a separate commercial 
# license must be obtained from swatah.ai (info@swatah.ai).
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
# for more details.
#
# Trademarks: All trademarks, service marks, and logos are the property of 
# their respective owners.

set -euo pipefail

# Initialize the current directory as a git repo and add EdgeYOLO as a submodule.
# Run this from the wrapper repo root if you started from an extracted zip.

if [ ! -d .git ]; then
  git init
  git branch -M main
fi

mkdir -p third_party

if [ ! -d third_party/edgeyolo/.git ]; then
  git submodule add https://github.com/LSH9832/edgeyolo.git third_party/edgeyolo
else
  echo "Submodule already exists at third_party/edgeyolo"
fi

git submodule update --init --recursive

echo "Done. EdgeYOLO submodule is available at third_party/edgeyolo"
