<!-- toc -->

   * [Installing](#installing)
      * [A note about rebooting](#a-note-about-rebooting)
      * [Step 1. Installing system-level dependencies (before accelerator-level dependencies)](#step-1-installing-system-level-dependencies-before-accelerator-level-dependencies)
         * [Installing dependencies on Ubuntu (before accelerator-level)](#installing-dependencies-on-ubuntu-before-accelerator-level)
      * [Step 2. Installing accelerator-level dependencies](#step-2-installing-accelerator-level-dependencies)
         * [Installing TTKMD (kernel-mode driver)](#installing-ttkmd-kernel-mode-driver)
         * [Installing tt-flash firmware](#installing-tt-flash-firmware)
         * [Installing tt-smi](#installing-tt-smi)
      * [Step 3. Installing system-level dependencies (after accelerator-level dependencies)](#step-3-installing-system-level-dependencies-after-accelerator-level-dependencies)
         * [Installing dependencies on Ubuntu (after accelerator-level)](#installing-dependencies-on-ubuntu-after-accelerator-level)
      * [Step 4. Installing developer dependencies](#step-4-installing-developer-dependencies)
         * [Installing developer-level dependencies on Ubuntu](#installing-developer-level-dependencies-on-ubuntu)
         * [About wheel installation](#about-wheel-installation)
      * [From source](#from-source)
   * [Getting started](#getting-started)
      * [Environment setup](#environment-setup)
      * [Running example programs](#running-example-programs)
      * [C++ Integration Tests](#c-integration-tests)
      * [Python Integration Tests](#python-integration-tests)
   * [Documentation](#documentation)
   * [Troubleshooting and debugging tips](#troubleshooting-and-debugging-tips)
   * [Contributing](#contributing)
   * [Communication](#communication)

<!-- tocstop -->

Table of Contents generated with
[github-markdown-toc](https://github.com/ekalinin/github-markdown-toc).

## Installing

**Note**: Currently, all features are only fully tested on Grayskull E150
accelerators. We are currently working on functionality for other Tenstorrent
architectures.

If you want to run this software on a Tenstorrent cloud machine, you can provision your own machine on the Tenstorrent cloud using documentation
[here](https://github.com/tenstorrent-metal/metal-internal-workflows/wiki/Installing-Metal-development-dependencies-on-a-TT-Cloud-VM).

These are the official ways of installing this software:

- [From source](#from-source)

However, you must have the appropriate accelerator-level and related
system-level dependencies. Otherwise, you may skip to your preferred
installation method in the above list.

### A note about rebooting

The full installation of accelerator-level and some system-level dependencies to use this software will require
a large number of reboots.

The minimum number of reboots you will require will be 2, for

- Installing the kernel-mode driver.
- Installing the first pass of hugepages changes.

If you're using a Grayskull card, flashing the firmware with the required
version will require another reboot. Wormhole doesn't have this requirement as
you can use `tt-smi` to reset your card.

If you're doing a full install on a Tenstorrent cloud machine and are planning
to install WekaFS to use models along with the hugepages changes required to
use WekaFS, you will require at least 2 more additional reboots. Because of the
indeterminate nature of WekaFS, you may require more.

### Step 1. Installing system-level dependencies (before accelerator-level dependencies)

System-level dependencies include the third-party libraries and hugepages settings. We have split this section into two parts. This is because you will require some of the accelerator-level dependencies to continue installing the system-level dependencies after the initial set.

#### Installing dependencies on Ubuntu (before accelerator-level)

1. Install some system-level dependencies through `apt`.

First, perform an update and install the dependencies:

```
sudo apt update
sudo apt install software-properties-common=0.99.9.12 build-essential=12.8ubuntu1.1 python3.8-venv=3.8.10-0ubuntu1~20.04.9 libgoogle-glog-dev=0.4.0-1build1 libyaml-cpp-dev=0.6.2-4ubuntu1 libboost-all-dev=1.71.0.0ubuntu2 libsndfile1=1.0.28-7ubuntu0.2 libhwloc-dev
```

2. Now continue to following sections to [install](#installing-accelerator-level-dependencies) accelerator-level dependencies and then the [required](#installing-system-level-dependencies-after-accelerator-level-dependencies) system-level dependencies that require the driver.

### Step 2. Installing accelerator-level dependencies

You must have the following accelerator-level dependencies:

For Grayskull:

- At least 1 unharvested E150 attached to a PCIe x16 slot
- TTKMD (Tenstorrent kernel-mode driver) v1.26
- ``tt-flash`` acclerator firmware fw_pack-80.4.0.0_acec1267.tar.gz
- ``tt-smi`` tt-smi_2023-06-16-0283a02404487eea or above

For Wormhole B0:

- At least 1 N150 or N300 attached via PCIe
- TTKMD (Tenstorrent kernel-mode driver) v1.26
- ``tt-flash`` acclerator firmware 2023-08-08 (7.D)
- ``tt-smi`` tt-smi-8.6.0.0_2023-08-22-492ad2b9ef82a243 or above

The instructions for installing TTKMD, `tt-flash`, and `tt-smi` follow.

#### Installing TTKMD (kernel-mode driver)

Please refer to the Tenstorrent [tt-kmd](https://github.com/tenstorrent/tt-kmd) page to get the specific version you need and install.

#### Installing `tt-flash` firmware

Please refer to the Tenstorrent [tt-flash](https://github.com/tenstorrent/tt-flash) page to get the tt-flash tool.

You will need to flash your accelerator with the specific version of firmware blob you are looking to install.

The firmware blob for Grayskull should be available [here](https://github.com/tenstorrent/tt-firmware-gs).

#### Installing `tt-smi`

Please refer to the Tenstorrent [tt-smi](https://github.com/tenstorrent/tt-smi) page to get the specific version you need and install.

If you are a developer, you should also go through the [section](#installing-developer-dependencies), in addition to any system-level dependencies required after these accelerator-level dependencies.

### Step 3. Installing system-level dependencies (after accelerator-level dependencies)

#### Installing dependencies on Ubuntu (after accelerator-level)

1. Download the raw latest version of the `setup_hugepages.py` script. It should be located [in the repository](https://github.com/tenstorrent-metal/tt-metal/blob/main/infra/machine_setup/scripts/setup_hugepages.py).

2. Invoke the first pass of the hugepages script and then reboot.

```
sudo -E python3 setup_hugepages.py first_pass && sudo reboot now
```

3. Invoke the second pass of the hugepages script and then check that hugepages is correctly set.

```
sudo -E python3 setup_hugepages.py enable && sudo -E python3 setup_hugepages.py check
```

**NOTE**: You may have to repeat the hugepages steps upon every reboot, depending on your system and other services that use hugepages.

4. If you are a developer, you should also go through the [section](#installing-developer-dependencies) on developer dependencies, in addition to accelerator-level dependencies.

### Step 4. Installing developer dependencies

#### Installing developer-level dependencies on Ubuntu

1. Install system-level dependencies for development through `apt`.

```
sudo apt install clang-6.0=1:6.0.1-14 git git-lfs cmake=3.16.3-1ubuntu1.20.04.1 pandoc libtbb-dev libcapstone-dev pkg-config
```

2. Download and install [Doxygen](https://www.doxygen.nl/download.html), version 1.9 or higher, but less than 1.10.

3. Download and install [gtest](https://github.com/google/googletest) from source, version 1.13, and no higher.

4. If you are working on experimental, internal model development, you must now also install and mount WekaFS. Note that this is only available on Tenstorrent cloud machines. The instructions are on this [page](https://github.com/tenstorrent-metal/metal-internal-workflows/wiki/Installing-Metal-development-dependencies-on-a-TT-Cloud-VM), which are only available to those who have access to the Tenstorrent cloud. Otherwise, you may skip this step if you are not working on such models. If you are a regular user of this software, you do not need WekaFS.

#### About wheel installation

We currently do not support installing our software from a wheel. Not all
features have been tested. The wheel is not an official release asset.

You can reference interim notes about wheel installation in
[documentation](infra/README_WHEELS.md) within source.

### From source

Currently, the best way to use our software is building from source.

You must also ensure that you have all accelerator-level and system-level
dependencies as outlined in the instructions above. This also includes
developer dependencies if you are a developer.


1. Clone the repo. If you're using a release, please use ``--branch
   <VERSION_NUMBER>``.

``<VERSION_NUMBER>`` is the release version you will be using. Otherwise, you can use ``main`` to get the latest development source.

```
git clone git@github.com:tenstorrent-metal/tt-metal.git --recurse-submodules --branch <VERSION_NUMBER>
cd tt-metal
```

For example, if you are trying to use version `v0.35.0`, you can execute:

```
git clone git@github.com:tenstorrent-metal/tt-metal.git --recurse-submodules --branch v0.35.0
cd tt-metal
```

Note that we also recommend you periodically check LFS and pull its objects
for submodules.

```
git submodule foreach 'git lfs fetch --all && git lfs pull'
```

2. Set up the environment. Note that this setup is required **every time you
   want to use this project**.

```
export ARCH_NAME=<arch name>
export TT_METAL_HOME=<this repo dir>
export PYTHONPATH=<this repo dir>:$TT_METAL_HOME/ttnn
export TT_METAL_ENV=dev
```

3. Build the project and activate the environment.

```
make build && source build/python_env/bin/activate
```

You should look ahead to [Getting started](#getting-started) to further use
this project.

## Getting started

### Environment setup

If you just came reading from building from source, you can read ahead to
[running an example](#running-example-programs).

Otherwise, you must set up the necessary environment variables to use this
project every time:

```
export ARCH_NAME=<arch name>
export TT_METAL_HOME=<appropriate value based on installation method above>
```

where ``<arch name>`` is your target, which could be:

- ``grayskull``
- ``wormhole_b0``

etc...

If you're setting up an environment from source, you must further set up and
activate the environment with:

```
export PYTHONPATH=<this repo dir>:$TT_METAL_HOME/ttnn
export TT_METAL_ENV=dev
source build/python_env/bin/activate
```

### Running example programs

After installing, please refer to our [Getting Started
page](https://tenstorrent-metal.github.io/tt-metal/latest/get_started/get_started.html)
in our documentation.

Note that example programs are only available through source installation at
this time.

### C++ Integration Tests

1. Build the API integration tests using the make command,
```
make tests
```
2. Run the test binaries from the path **${TT_METAL_HOME}/build/test/tt_metal**

### Python Integration Tests

1. Initialize the Python virtual environment [see documentation](#Environment-setup)
2. Run the specific test point with pytest tool, e.g.
   ```
   $ pytest tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_composite.py
   ```
3. If you have any issues with import paths for python libraries include the following environment variable,
   ```
   $ export PYTHONPATH=${PYTHONPATH}:${TT_METAL_HOME}:${TT_METAL_HOME}/ttnn
   ```

## Documentation

Please refer to our
[documentation](https://tenstorrent-metal.github.io/tt-metal/latest/index.html).

## Troubleshooting and debugging tips

In addition to our documentation above, you can check out relevant sections in
the [contribution
standards](https://github.com/tenstorrent-metal/tt-metal/blob/main/CONTRIBUTING.md)
if you ever need hardware troubleshooting help or debugging tips.

## Contributing

We are excited to move our development to the public, open-source domain.
However, we are not adequately staffed to review contributions in an expedient
and manageable time frame at this time. In the meantime, please review the
[contributor's guide](CONTRIBUTING.md) for more information about contribution
standards.

## Communication

Announcements from the Tenstorrent team regarding this project will be in the
[discussions
page](https://github.com/tenstorrent-metal/tt-metal/discussions/categories/general-announcements).

We also have a Discord channel that you can join. You may discuss with other
members of the community and developers there. You may use this invite
[link](https://discord.gg/tvhGzHQwaj).

If you would like to formally propose a new feature, report a bug, or have
issues with permissions, please file through [GitHub
issues](https://github.com/tenstorrent-metal/tt-metal/issues/new/choose).
