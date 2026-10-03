#!/usr/bin/env bash
# Build the Canvas submission zip:  ./package.sh Lastname   ->  Lastname-hw2.zip
#   Lastname-hw2/{src/, runs/, plots.xlsx, plots.pdf, ai/}
set -euo pipefail
cd "$(dirname "$0")"
name=${1:?usage: ./package.sh Lastname}
dir="$name-hw2"
rm -rf "$dir" "$dir.zip"
mkdir -p "$dir/src"
cp src/*.c src/*.h src/Makefile src/run_all.sh src/plots.py src/requirements.txt src/README.md "$dir/src/"
cp -r runs ai "$dir/"
cp plots.xlsx plots.pdf "$dir/" 2>/dev/null || echo "warning: plots not found; run src/run_all.sh first" >&2
if command -v zip >/dev/null; then
    zip -qr "$dir.zip" "$dir"
else
    python3 -c "import shutil; shutil.make_archive('$dir', 'zip', '.', '$dir')"
fi
rm -rf "$dir"
echo "wrote $dir.zip"
