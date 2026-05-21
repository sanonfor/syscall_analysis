#!/bin/bash
set -euo pipefail

IN_DIR=$1
OUT_DIR=uniq_data

for f in $(find $IN_DIR -type f); do
	mkdir -p $OUT_DIR/$(dirname $f)
	echo $f
	uniq $f > $OUT_DIR/$f
done
