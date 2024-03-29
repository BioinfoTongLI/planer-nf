#!/usr/bin/env/ nextflow

// Copyright (C) 2020 Tong LI <tongli.bioinfo@protonmail.com>

nextflow.enable.dsl=2

params.out_dir = "/tmp/nuc_seg/" //output dir
params.img = "DAPI.tif" // path to DAPI tif
params.scale = 0.25

params.model_dir = projectDir + "/models"
/*params.container = "hamat/planer_gpu:cuda114"*/
params.container = "/lustre/scratch117/cellgen/team283/tl10/sifs/planer.sif"

process Compute_flow {
    cache "lenient"
    container params.container
    containerOptions "--nv -B ${params.model_dir}:/model:ro"
    publishDir params.out_dir, mode: "copy"

    /*maxForks 1*/
    clusterOptions = "-gpu 'num=1:gmem=50000' -m dgx-b11"

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
    containerOptions "--nv"
    /*storeDir params.out_dir*/
    publishDir params.out_dir, mode: "copy"

    clusterOptions = "-gpu 'num=1:gmem=50000' -m dgx-b11"

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


process nuclei_seg {
    echo true
    container "/lustre/scratch117/cellgen/team283/tl10/sifs/cellpose_2.0.sif"
    containerOptions "--nv"
    /*containerOptions "--gpus all"*/
    clusterOptions = "-gpu 'num=1:gmem=31000' -m dgx-b11"
    storeDir params.out_dir

    memory { 680.GB * task.attempt }
    /*time { 1.hour * task.attempt }*/

    errorStrategy { task.exitStatus in 137..140 ? 'retry' : 'terminate' }
    maxRetries 2

    input:
    path(ome_tif)

    output:
    tuple val(stem), path("${stem}_nuclei_seg.tif")

    script:
    stem = file(ome_tif.baseName).baseName
    """
    nuclei_seg.py "${ome_tif}" --stem "${stem}" --diam 45
    """
}


workflow {
    Compute_flow(Channel.fromPath(params.img), params.scale)
    Post_process(Compute_flow.out)
}

workflow cellpose {
    nuclei_seg(Channel.fromPath(params.img))
}
