#!/bin/sh

set -x

protoc protos/*.proto --python_out=. || exit -1

exit 0
