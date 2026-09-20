#!/usr/bin/env bash
# Make `mjpython` work inside a uv-created venv on macOS.
#
# MuJoCo's interactive viewer must be launched with `mjpython` on macOS.
# mjpython is a signed app bundle that dlopen()s the venv's python, which in
# turn needs libpython3.11.dylib. uv's standalone CPython ships that dylib but
# does not put it on the rpath, so the load fails with:
#     Library not loaded: @rpath/libpython3.11.dylib
# DYLD_LIBRARY_PATH does not help -- macOS strips DYLD_* for signed binaries.
# But one of the paths mjpython searches is <venv>/bin/../, so putting a copy
# there fixes it.
set -euo pipefail
VENV="${1:-.venv}"
PYLIB="$("$VENV/bin/python" -c 'import sysconfig,os; print(os.path.join(sysconfig.get_config_var("installed_base"),"lib"))')"
VER="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SRC="$PYLIB/libpython$VER.dylib"
[ -f "$SRC" ] || { echo "no $SRC -- your Python has no shared libpython"; exit 1; }
cp "$SRC" "$VENV/libpython$VER.dylib"
echo "copied $SRC -> $VENV/libpython$VER.dylib"
echo "now run:  $VENV/bin/mjpython play.py"
