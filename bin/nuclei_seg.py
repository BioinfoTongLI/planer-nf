#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Tong LI <tongli.bioinfo@protonmail.com>
#
# Distributed under terms of the BSD-3 license.

"""

"""
import fire
from cellpose import models
import tifffile as tf
from dask_image.imread import imread
import numpy as np


def main(ome_tif, stem, diam):
    dapi = imread(ome_tif)[0]
    model = models.Cellpose(gpu=True, model_type='cyto2')
    # model = models.Cellpose(gpu=False, model_type='cyto2')
    masks, flows, styles, diams = model.eval(dapi,
            diameter=diam, flow_threshold=None, channels=[0, 0])
    tf.imwrite(f"{stem}_nuclei_seg.tif", masks)

if __name__ == "__main__":
    fire.Fire(main)
