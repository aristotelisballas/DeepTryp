# Physionet 2022 challenge

> DeepTryp


The root folder (the one where this file is)
contains the code provided by the challenge organizers. All team code is placed
in the `teamcode` folder. 


## File list

* `Dockerfile`: this is the original Dockerfile, modified to use the latest
  Tensorflow image with GPU capabilities.
* `env.sh`: defines useful variables for the `Makefile` and `run-docker.sh`
  files.
* `Makefile`: the Makefile for building the docker image (also removes the
  previous image)
* `requirements.txt`: a simple requirements file that is used by the
  `Dockerfile` when setting up the image. If you get an error the first time,
  because there is no image to be removed, use `make build` to only build the
  image.
* `run-docker.sh`: convenience script for starting an container with the built
  image and launching an interactive bash session.
* `*.py`: the original code for training, testing and evaluating, as provided by
  the challenge organizers. 


## How to use

Assuming you have a workstation with NVIDIA GPUs and working installation of
NVIDIA drivers, [docker](https://www.docker.com/), and the [NVIDIA
docker](https://github.com/NVIDIA/nvidia-docker), and you have cloned this
repository, the following helps you run the code in the container.

### Building the image

First, build the image:

```bash
sudo make build
```

Note that `sudo` is required as `docker` only runs as root. After the build
finishes, the image is available in your local docker image repo, and you can
see it with:

```bash
sudo docker image ls
```

Note that every time you change your code, you need to rebuild the docker, as
the code is "packaged" within the docker image. To remove the existing image
when you rebuild, you can firt run:

```bash
sudo make clean
```

before building, or simply run:

```bash
sudo make all
```

to both `clean` and `build`.


### Starting the container

To start a one-off container, you can use the convenience script
`run-docker.sh`. First you should edit and update the path variables that
are defined in `env.sh`. 

Then, simply run:

```bash
sudo ./run-docker.sh
```

to start an interactive bash session inside the container.

### Running inside the container

Inside inside the container, you can use the following commands for training,
running, and evaluating the model:

```bash
python train_model.py -d training_data -m model -v
python run_model.py -d holdout_data -m model -o holdout_outputs -v
python evaluate_model.py -d holdout_data -o holdout_outputs
```

