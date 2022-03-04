#!/usr/bin/env/ nextflow

// Copyright (C) 2020 Tong LI <tongli.bioinfo@protonmail.com>

nextflow.enable.dsl=2

params.out_dir = "/tmp/test" //output dir
params.img = "/home/ubuntu/Documents/playground/out_opt_flow_registered_DAPI.tif" // path to DAPI tif

params.model_dir = projectDir + "/models"
params.container = "hamat/planer_gpu:cuda114"

process Compute_flow {
    cache "lenient"
    container params.container
    containerOptions "--gpus all -v ${params.model_dir}:/model:ro"
    publishDir params.out_dir, mode: "copy"

    input:
    path(img)

    output:
    tuple val(stem), path("*flow.npy")

    script:
    stem = img.baseName
    """
    compute_flow.py --stem ${stem} --img_p ${img}
    """
}

process Post_process {
    cache "lenient"
    container params.container
    /*containerOptions "--gpus all"*/
    /*storeDir params.out_dir*/
    publishDir params.out_dir, mode: "copy"

    input:
    tuple val(stem), path(img)

    output:
    path("*lab.tif")

    script:
    """
    post_process.py --stem $stem --flow_npy ${img}
    """
}

workflow {
    Compute_flow(Channel.fromPath(params.img))
    Post_process(Compute_flow.out)
}
