#! /bin/sh
#
# run.sh
# Copyright (C) 2021 Tong LI <tongli.bioinfo@gmail.com>
#
# Distributed under terms of the BSD-3 license.
#



NXF_OPTS='-Dleveldb.mmap=false' nextflow run /lustre/scratch117/cellgen/team283/tl10/planer-nf/main.nf \
	-params-file $1 \
	-profile standard,local \
	-entry cellpose \
	-resume
