#!/usr/bin/env/ nextflow

// Copyright (C) 2020 Tong LI <tongli.bioinfo@protonmail.com>

nextflow.enable.dsl=2

params.out_dir = "/tmp/nuc_seg/" //output dir
params.img = "DAPI.tif" // path to DAPI tif
params.scale = 0.25

params.model_dir = projectDir + "/models"
params.container = "hamat/planer_gpu:cuda114"

process Compute_flow {
    cache "lenient"
    container params.container
    containerOptions "--gpus all -v ${params.model_dir}:/model:ro"
    publishDir params.out_dir, mode: "copy"

    /*maxForks 1*/

    input:
    path(img)
    val(scale)

    output:
    tuple val(stem), path("*flow.npy")

    script:
    stem = img.baseName
    """
    compute_flow.py --stem "${stem}" --img_p ${img} --scale ${scale}
    """
}

process Post_process {
    cache "lenient"
    container params.container
    containerOptions "--gpus all"
    /*storeDir params.out_dir*/
    publishDir params.out_dir, mode: "copy"

    maxForks 1

    input:
    tuple val(stem), path(img)

    output:
    path("*lab.tif")

    script:
    """
    flow_tile.py --stem "$stem" --flow_npy ${img}
    """
}

workflow {
    Compute_flow(Channel.fromPath(params.img), params.scale)
    Post_process(Compute_flow.out)
}
