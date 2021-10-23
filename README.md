# Instance Segmentation & Transfer Learning
## _Seabirds Master Thesis_
# _Alternative Solution to Catastrophical Forgetting on Few-Shot Instance Segmentation_

This README file contains technichal information about the thesis performed within AI Sweden, SLU and KTH University in 2021.

## Requirements

- Docker. Refactoring needed for *.sh files
- Datasets YTVIS and Seabirds
- YTVIS dataset must contain a train and test folder per video. References must be updated.
- Detectron2 correctly installed
- NVIDIA DGX A100 GPU unit 0
- Compatibel trained model when using inference setup

## Features

- Easy installation running docker
- TODO

## Running

This software allows to easily run using Docker engine. To run Docker container only executing the .sh files is needed. It is required to refactor the .sh files.
1. Build image. Easily done by running:
```sh
sh build.sh
````
2. Run image. Two options, detached or interactive mode. Detached mode recommended for experimentation and training. Interactive mode recommended for debugging and developping.
```sh
sh run_it.sh
sh run_d.sh
```
3. Results stored in output. 

By default output is gitignored. Keep in mind models are heavy files, making versioning and uploading to GitHub not ideal.


## Developing

Code is found in src folder. It must be a volume, same as volume (refer to sh run*). File commands.sh contains the commands to be executed for every run in detached mode. This file name can be changed in Dockerfile and then rebuilding is required.

src file contains thesis-related python scripts and the necessary script to train the seabirds dataset. 

By default, seabirds dataset will be trained. This can be done by running data_processing_seabirds.py 
data_processing.py containes the code to train a two-stage transfer learning model based on the YTVIS dataset. Keep in mind folder strucutre expected to do so.

inference.py runs testing on YTVIS dataset. It can be edited to run inference with the seabirds model. By default, data_processing_seabirds.py automatically not only trains but also runs inference on a randomly assigned 500 images sample from sequence_sampled folder.

Data can be found in data folder within juan.vallado user folder.






## License

MIT

## Reference
TODO: Add thesis paper.


