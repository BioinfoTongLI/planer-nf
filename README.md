# planer-nf

A [nextflow](https://www.nextflow.io/) pipeline that wraps [planer](https://github.com/Image-Py/planer)-[cellpose](https://www.cellpose.org/) to do single nuclei segmentation.

# Installation

1. Make sure you've installed Java and nextflow following the [official instruction](https://www.nextflow.io/index.html#GetStarted)
2. Make sure you've installed [Docker](https://www.docker.com/)
3. Create your config file following the [template.yaml](https://github.com/BioinfoTongLI/planer-nf/blob/main/template.yaml)
4. Run with `nextflow run main.nf -params-file [path-to-your.yaml]`
