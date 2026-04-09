#!/usr/bin/env bash
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
