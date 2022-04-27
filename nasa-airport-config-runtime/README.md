# Run-way Functions: Prediction Reconfigurations at US Airports

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue) [![GPU Docker Image](https://img.shields.io/badge/Docker%20image-gpu--latest-green)](https://hub.docker.com/r/drivendata/nasaairportconfig-competition/tags?page=1&name=gpu-latest) [![CPU Docker Image](https://img.shields.io/badge/Docker%20image-cpu--latest-green)](https://hub.docker.com/r/drivendata/nasaairportconfig-competition/tags?page=1&name=cpu-latest)


### For instructions about how to submit to the [Run-way Functions Challenge](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened), start with the [Code submission format page](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/page/442/) of the competition website.

Welcome to the runtime repository for the [Run-way Functions: Predict Reconfigurations at US Airports](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/)! This repository contains the definition of the environment where your code submissions will run. It specifies both the operating system and the software packages that will be available to your solution.

This repository has three primary uses for competitors:

:bulb: **Provide example solutions**: You can find two examples to help you develop your solution. 
1. [Baseline solution](https://github.com/drivendataorg/nasa-airport-config-runtime/tree/main/submission_src): minimal code that runs succesfully in the runtime environment output and outputs a proper submission. This simply predicts equal probabilities for each configuration at an airport. You can use this as a guide to bring in your model and generate a submission.
2. Implementation of the [benchmark solution](https://github.com/drivendataorg/nasa-airport-config-runtime/tree/main/benchmark_src): submission code based on the [benchmark blog post](https://www.drivendata.co/blog/airport-configuration-benchmark/).

:wrench: **Test your submission**: Test your submission using a locally running version of the competition runtime to discover errors before submitting to the competition site. You can also find an [scoring script](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/scripts/score.py) implementing the competition metric.

:package: **Request new packages in the official runtime**: Since your submission will not be able to access the internet, all packages must be pre-installed. If you want to use a package that is not in the runtime environment, make a pull request to this repository. Make sure to test out adding the new package to both official environments, [CPU](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/environment-cpu.yml) and [GPU](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/environment-gpu.yml).

----


### [0. Getting started](#getting-started)
 - [Prerequisites](#prerequisites)
 - [Simulating test data](#simulating-test-data)
 - [Submission format](#submission-format)
### [1. Testing a submission locally](#testing-a-submission-locally)
 - [Running your submission locally](#running-your-submission-locally)
 - [Scoring your predictions](#scoring-your-predictions)
 - [Running the benchmark](#running-the-benchmark)
### [2. Troubleshooting](#troubleshooting)
 - [Downloading pre-trained weights](#downloading-pre-trained-weights)
 - [CPU and GPU](#cpu-and-gpu)
### [3. Updating runtime packages](#updating-runtime-packages)

----

## Getting started

### Prerequisites

 - A clone or fork of this repository
 - [Docker](https://docs.docker.com/get-docker/)
 - At least ~12 GB of free space for both the sample data and Docker images
 - [GNU make](https://www.gnu.org/software/make/) (optional, but useful for running the commands in the Makefile)

Additional requirements to run with GPU:

 - [NVIDIA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) with **CUDA 11**
 - [NVIDIA Docker container runtime](https://nvidia.github.io/nvidia-container-runtime/)

### Simulating test data

To run a submission, the code execution environment reads features and the submission format from the `runtime/data` directory on the host machine.

```
$ tree runtime/data
├── katl
│   ├── katl_airport_config.csv.bz2
│   ├── katl_arrival_runway.csv.bz2
│   ├── ...
│   └── katl_tfm_estimated_runway_arrival_time.csv.bz2
├── kclt
│   ├── kclt_airport_config.csv.bz2
│   ├── kclt_arrival_runway.csv.bz2
│   ├── ...
│   └── kclt_tfm_estimated_runway_arrival_time.csv.bz2
│   ...
├── ksea
│   ├── ksea_airport_config.csv.bz2
│   ├── ksea_arrival_runway.csv.bz2
│   ├── ...
│   └── ksea_tfm_estimated_runway_arrival_time.csv.bz2
└── submission_format.csv
```

We do not provide the full prescreened test set, so instead we need to simulate that data. We provide two versions:

#### A small fake dataset

This is a small dataset of not-very-realistic data suitable for quickly making sure that your code runs. Each feature column is simulated by resampling 10 of its values (with replacement) 100 times. You can generate the fake dataset by running:

The script [`scripts/generate_fake_dataset.py`](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/scripts/generate_fake_dataset.py) generates the development dataset. You can learn more about it with the `--help` flag:

```
$ python scripts/generate_fake_dataset.py --help

Usage: generate_fake_dataset.py [OPTIONS] [FAKE_FEATURE_PARAMS_PATH]
                                [SUBMISSION_FORMAT_PATH]

  Generates a small, not very realistic data that nonetheless can stand in
  as the features and submission format for testing out the code execution
  submission.

  Reads fake data parameters where each feature CSV is described by the
  column names and 5 sample values for each column. 100 rows are simulated
  by sampling (with replacement) from those sample values. The start time
  for the fake data is the first time in the submission format, and the end
  time is the last time in the submission format.

Arguments:
  [FAKE_FEATURE_PARAMS_PATH]  Path to a JSON file with fake data parameters.
                              [default: scripts/fake_feature_params.json]

  [SUBMISSION_FORMAT_PATH]    Path to a sample submission format.  [default:
                              scripts/fake_submission_format.csv.bz2]


Options:
  --output-directory PATH  Directory where the development dataset will be
                           saved.  [default: runtime/data]

  --seed INTEGER           [default: 10]
  --help                   Show this message and exit.
```

By default, the fake dataset will be saved to `runtime/data`, which is where the later steps expect it to be.

#### A larger development dataset

The development dataset takes a subset of data from the open training dataset; labels from the final day and features from the final day plus a _warm start_ period that starts two days prior to the final day. It requires that you have the open training features and submission format downloaded locally. (You can download them from the [Data download page](https://www.drivendata.org/competitions/89/competition-nasa-airport-configuration/data/).) The development dataset is suitable for a more realistic test of your submission. If your submission runs successfully on the development dataset, it _should_ run successfully on the prescreened dataset.

The script [`scripts/generate_development_dataset.py`](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/scripts/generate_development_dataset.py) generates the development dataset. By default, the script assumes that the features and labels are located in the `data` directory in the root of this repository, but you can specify another location. You can learn more about it with the following command:

```
$ python scripts/generate_development_dataset.py --help
Usage: generate_development_dataset.py [OPTIONS] [INPUT_FEATURE_DIRECTORY]
                                       [INPUT_LABELS_PATH] [OUTPUT_DIRECTORY]

  Creates a data subset that matches the format of the full evaluation data
  intended for developing your submission.

Arguments:
  [INPUT_FEATURE_DIRECTORY]  Directory containing the training features.
                             [default: data]

  [INPUT_LABELS_PATH]        Path to the training labels.  [default:
                             data/open_train_labels.csv.bz2]

  [OUTPUT_DIRECTORY]         Directory where the development dataset will be
                             saved.  [default: runtime/data]


Options:
  --help  Show this message and exit.
```

The default values assume that the train features and labels are in the `data` directory in the root of this repository and will save the development dataset to `runtime/data`.

When testing your submission locally, a Docker container is launched from your computer (the "host" machine), and the `runtime/data` directory on your host machine is mounted as a read-only directory to `codeexecution/data`. In the runtime, your code will then be able to access test features from `codeexecution/data/test_features`.

For details about how data is accessed in the code execution runtime, see the [Code submission format page](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/page/442/).

### Code submission format

Time is a key element in this competition―we're interested in a _real-time_ solution, one that can predict the future using only information available at the present. Assuring that a solution doesn't (accidentally or otherwise) use information from the future is complicated since the final evaluation dataset is a static dataset containing many different prediction times; a feature at 9 AM is valid for predicting 10 AM (it's in the past) but invalid for predicting 8 AM (it's in the future). In other words, each prediction time defines a unique set of valid features, different from that of all other prediction times! This makes it challenging to ensure that your submission only uses valid time points for each prediction time (and even more challenging for the competition hosts to validate that _all_ submissions are valid!).

The code execution runtime is designed to avoid the need to track valid and invalid features. It simulates real-time conditions for each prediction time and provides a simple way to access features that are guaranteed to be in the past. The process is defined in the [`runtime/entrypoint.sh`](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/entrypoint.sh), which runs the submissions. To summarize, for each prediction time in the submission format:

1. The [**supervisor**](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/submission-service-alternating/runtime/supervisor.py) script creates a time-censored extract of the features, i.e., features _prior to_ the prediction time, and stores them to `/codeexecution/data` with the same directory structure as the training features. It also creates a **partial submission format**, `/codeexecution/data/partial_submission_format.csv`, which includes only rows for the current prediction time.
2. Your `main.py` runs with a single argument, the prediction time, e.g., `2022-06-09T14:00:00`. It can read your model assets, any or all of the time-censored features in `/codeexecution/data`, the partial submission format, and any intermediate files you may have written out in previous iterations. It must write a CSV to `/codeexecution/prediction.csv` that includes predicted probabilities for all airports, configurations, and lookaheads for the input prediction time. The CSV should have the same format as the partial submission format with your predicted probabilities in the `active` column.
3. A utility script checks that your predictions for this prediction time has the same indices and same columns as the partial submission format and that the probabilities for each airport and lookahead sum to 1.

The process is repeated for each prediction time in order, and then the individual predictions are combined into a single CSV, which is saved to `/codeexecution/submission/submission.csv`.

So as long as you only read from `/codeexecution/data`, you can be sure your model isn't using information from the future. We provide the supervisor and utility scripts; all you need to do is provide `main.py`. We have included two sample solutions to help you get started:
- [`submission_src/main.py`](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/submission_src/main.py), which implements the simplest possible valid submission, using the submission format (uniform probabilties across all configurations) as its prediction.
- [`benchmark_src/main.py`](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/benchmark_src/main.py), which implements the _Recency-weighted historical forecast_ model from the [benchmark blog post](https://www.drivendata.co/blog/airport-configuration-benchmark).

Here's a few **do**s and **don't**s:

**Do**:
- Write a script `main.py` implementing a command line interface (CLI) that takes a single positional argument the current prediction time provided in the [datetime format](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) `%Y-%m-%dT%H:%M:%S`, e.g., `2021-10-20T23:00:00`.
- Read features from `/codeexecution/data`. These are guaranteed to be in the past relative to the prediction time.
- Write your prediction for all lookaheads at the current prediction time to `/codeexecution/prediction.csv`.
- Write out and read intermediate files. Just be careful not to use any of the reserved file locations: `/codeexecution/prediction.csv`, `/codeexecution/submission/submission.csv`.

**Don't**:
- Read from locations other than `/codeexecution/data`, your model assets, and intermediate files that your submission produces. Participants who attempt to read features from unauthorized locations or otherwise circumvent the protections in place will be disqualified from the competition.


## Testing a submission locally

When you make a submission on the DrivenData competition site, we run your submission inside a Docker container, a virtual operating system that allows for a consistent software environment across machines. **The best way to make sure your submission to the site will run is to first run it successfully in the container on your local machine.**

For the full requirements of a submission, see the [Code submission format page](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/page/442/).


### Running your submission locally

This section provides instructions on how to run the your submission in the code execution container from your local machine. The basic steps are:

```
make pull
make pack-submission
make test-submission
```

Here's a bit more detail:

1. Set up the [prerequisites](#prerequisites).
2. [Simulate the test dataset](#simulating-test-data) in `runtime/data`.
3. Download the official competition Docker image:

To simplify the steps, key processes have been defined in the `Makefile`. Commands from the `Makefile` are then run with `make {command_name}`.

```
$ make pull
```

4. Save all of your submission files, including the required `main.py` script, in the `submission_src` folder of the runtime repository. Make sure any needed model weights and other assets are saved in `submission_src` as well.

5. Create a `submission/submission.zip` file containing your code and model assets:

```
$ make pack-submission
cd submission_src; zip -r ../submission/submission.zip ./*
  adding: main.py (deflated 50%)
```

6. Launch an instance of the competition Docker image, and run the same inference process that will take place in the official runtime:

```
$ make test-submission
```

This unzips `submission/submission.zip` in the root directory of the container, and then runs the container entrypoint [entrypoint](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/entrypoint.sh), which iterates over the dataset, runs your `main.py` at each iteration, and compiles the individual predicitions into a single CSV for scoring. The final submission is saved out to `submission/submission.csv` on your local machine.
   
> Remember that `codeexecution/data` is a mounted version of what you have saved locally in `runtime/data`. In the official code execution platform, `codeexecution/data` will contain the actual test features.
> Also be aware that some of the file restrictions in place in the actual code execution environment are not enforced in the local environment. For example, in the local environment, your code can access `/clouddata`, which is blocked in the actual code execution environment.

When you run `make test-submission` the logs will be printed to the terminal and written out to `submission/log.txt`. If you run into errors, use the `log.txt` to determine what changes you need to make for your code to execute successfully. For an example of what the logs look like when the full process runs successfully, see [`example_log.txt`](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/example_log.txt).

### Scoring your predictions

We have provided a [scoring script](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/scripts/score.py) to calculate the competition metric in the same way scores will be calculated in the DrivenData platform. Both the fake dataset and development dataset include test labels, allowing you to score your predictions (although the actual scores won't be very meaningful).

1. After running your submission, the predictions generated by your code should be saved to `submission/submission.csv`.
2. Make sure the simulated test labels are saved in `runtime/data/test_labels.csv`.
3. Run `scripts/score.py` on your predictions:

```
python scripts/score.py
2022-02-03 10:32:38.116 | INFO     | __main__:main:35 - Airport scores:
{
  "katl": 0.15841057013179075,
  "kclt": 0.2711893730418441,
  "kden": 0.11251593411963931,
  "kdfw": 0.14250586739273782,
  "kjfk": 0.2573186405438316,
  "kmem": 0.14250586739273782,
  "kmia": 0.15407610367102945,
  "kord": 0.12169240267806194,
  "kphx": 0.214559155176405,
  "ksea": 0.2868359830561606
}
2022-02-03 10:32:38.116 | SUCCESS  | __main__:main:37 - Score: 0.18616098972042383
```

For details about the scoring script, you can run:

```
$ python scripts/score.py --help
Usage: score.py [OPTIONS]

  Computes the mean-aggregated log loss for a set of predictions given some
  test labels.

Options:
  --predictions-path PATH  [default: submission/submission.csv]
  --labels-path PATH       [default: runtime/data/test_labels.csv]
  --help                   Show this message and exit.
```


### Running the benchmark

The code for the [benchmark](https://www.drivendata.co/blog/airport-configuration-benchmark/) is also provided as an example of how to structure a more complex submission. See the benchmark [blog post](https://www.drivendata.co/blog/airport-configuration-benchmark/) for a full walkthrough. The process to run the benchmark is the same as running your own submission, except that you will reference code in `benchmark_src` rather than `submission_src`.

```
make pull
make pack-benchmark
make test-submission
```

Note that here we are running `pack-benchmark` instead of `pack-submission`. Just like with your submission, the final predictions will be saved to `submission/submission.csv` on your local machine.

## Troubleshooting

### Downloading pre-trained weights

Fine-tuning an existing model is common practice in machine learning. Many software packages will download the pre-trained model from the internet behind the scenes when you instantiate a model. That will fail in the the code execution environment, since submissions do not have open access to the internet. Instead you will need to include all weights along with your `submission.zip` and make sure that your code loads them from disk and rather than trying to download them from the internet.

For example, PyTorch uses a local cache which by default is saved to `~/.cache/torch`. Identify which of the weights in that directory are needed to run inference (if any), and copy them into your submission. If we need pre-trained ResNet34 weights we downloaded from online, we could run:

```sh
# Copy your local pytorch cache into submission_src/assets
cp ~/.cache/torch/checkpoints/resnet34-333f7ec4.pth submission_src/assets/

# Zip it all up in your submission.zip
zip -r submission.zip submission_src
```

If we wanted to copy all of the contents in the PyTorch cache, we could instead run `cp -R ~/.cache/torch submission_src/assets/`. When the platform runs your code, it will extract `assets` to `/codeexecution/assets`. You'll need to tell PyTorch to use your custom cache directory instead of `~/.cache/torch` by setting the `TORCH_HOME` environment variable in your Python code (in `main.py` for example).

```python
import os
os.environ["TORCH_HOME"] = "/codeexecution/assets/torch"
```

Now PyTorch will load the model weights from the local cache, and your submission will run correctly in the code execution environment without internet access.

### CPU and GPU

The `make` commands will try to select the CPU or GPU image automatically by setting the `CPU_OR_GPU` variable based on whether `make` detects `nvidia-smi`.

**If you have `nvidia-smi` and a CUDA version other than 11**, you will need to explicitly set `make test-submission` to run on CPU rather than GPU. `make` will detect your GPU and automatically select the GPU image, but it will fail because `make test-submission` requires CUDA version 11. 
```bash
CPU_OR_GPU=cpu make pull
CPU_OR_GPU=cpu make test-submission
```

If you want to try using the GPU image on your machine but you don't have a GPU device that can be recognized, you can use `SKIP_GPU=true`. This will invoke `docker` without the `--gpus all` argument.

## Updating runtime packages

If you want to use a package that is not in the environment, you are welcome to make a pull request to this repository. If you're new to the GitHub contribution workflow, check out [this guide by GitHub](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

The runtime manages dependencies using [conda](https://docs.conda.io/en/latest/) environments. [Here is a good general guide](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533) to conda environments. The official runtime uses **Python 3.9.7** environments.

To submit a pull request for a new package:

1. Fork this repository.
   
2. Edit the [conda](https://docs.conda.io/en/latest/) environment YAML files, `runtime/environment-cpu.yml` and `runtime/environment-gpu.yml`. There are two ways to add a requirement:
    - Conda package manager: Add an entry to the `dependencies` section. This installs from a conda channel using `conda install`. Conda performs robust dependency resolution with other packages in the `dependencies` section, so we can avoid package version conflicts.
    - Pip package manager: Add an entry to the `pip` section. This installs from PyPI using `pip`, and is an option for packages that are not available in a conda channel.

For both methods be sure to include a version, e.g., `numpy==1.20.3`. This ensures that all environments will be the same.

3. Locally test that the Docker image builds successfully for CPU and GPU images:

```sh
CPU_OR_GPU=cpu make build
CPU_OR_GPU=gpu make build
```

4. Commit the changes to your forked repository.
   
5. Open a pull request from your branch to the `main` branch of this repository. Navigate to the [Pull requests](https://github.com/drivendataorg/nasa-airport-config-runtime/pulls) tab in this repository, and click the "New pull request" button. For more detailed instructions, check out [GitHub's help page](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).
   
6. Once you open the pull request, Github Actions will automatically try building the Docker images with your changes and running the tests in `runtime/tests`. These tests can take up to 30 minutes, and may take longer if your build is queued behind others. You will see a section on the pull request page that shows the status of the tests and links to the logs.
   
7. You may be asked to submit revisions to your pull request if the tests fail or if a DrivenData team member has feedback. Pull requests won't be merged until all tests pass and the team has reviewed and approved the changes.

---

## Good luck; have fun!

Thanks for reading! Enjoy the competition, and [hit up the forums](https://community.drivendata.org/c/runway/52) if you have any questions!
