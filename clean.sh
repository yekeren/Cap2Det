#!/bin/sh

set -x


rm -f protos/*_pb2.py
find . -maxdepth 2 -type f -name "*.swo" | xargs rm -f
find . -maxdepth 2 -type f -name "*.swp" | xargs rm -f
find . -maxdepth 2 -type f -name "*.swx" | xargs rm -f

exit 0
