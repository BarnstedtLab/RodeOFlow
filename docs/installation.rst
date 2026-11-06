Installation
============

RodeOFlow can be installed in a Conda environment on Linux or Windows. A
Docker route for NVIDIA GPU-accelerated Farneback optical flow is also being
prepared.

Requirements
------------

For the Conda installation, you need Git and Conda (Miniforge or Anaconda).

Create the Conda environment
----------------------------

Clone the repository and enter its folder:

.. code-block:: bash

   git clone https://github.com/BarnstedtLab/RodeOFlow.git
   cd RodeOFlow

On Linux:

.. code-block:: bash

   conda env create -f environment_setup_linux.yaml
   conda activate rodeoflow

On Windows:

.. code-block:: bash

   conda env create -f environment_setup_windows.yaml
   conda activate rodeoflow

Install DeepLabCut
------------------

RodeOFlow uses DeepLabCut with its PyTorch backend. If DeepLabCut is not
already installed in the environment, install the pre-release version:

.. code-block:: bash

   python -m pip install --pre deeplabcut

Install RodeOFlow
-----------------

Install RodeOFlow from GitHub:

.. code-block:: bash

   python -m pip install "git+https://github.com/BarnstedtLab/RodeOFlow.git" --no-deps

The ``--no-deps`` option assumes that you created the Conda environment above
and installed DeepLabCut. If you use an existing environment, install the
dependencies required by the tracking and optical-flow engines you plan to
use. Installing RodeOFlow without ``--no-deps`` installs the dependencies
listed in the package metadata, but does not replace the separate
DeepLabCut, PyTorch, or CUDA-enabled OpenCV setup.

Dense optical flow options
--------------------------

The CUDA-accelerated Farneback option requires a compatible NVIDIA driver and
an OpenCV build compiled with CUDA support. Installing CUDA-enabled PyTorch
does not enable CUDA in OpenCV.

If you do not have a compatible CUDA-enabled OpenCV build, use DIS optical
flow on the CPU.

Optional: Docker with GPU-accelerated Farneback
-----------------------------------------------

A RodeOFlow Docker image with CUDA-enabled OpenCV is being prepared. It is
intended for users who want GPU-accelerated Farneback without building
CUDA-enabled OpenCV themselves.

GPU access from Docker requires an NVIDIA GPU and compatible driver on the
host, Docker, and the NVIDIA Container Toolkit. The container must be started
with GPU access enabled, typically using the ``--gpus all`` option. The
container image must include the CUDA-enabled OpenCV build.

The image name, tested platform requirements, and exact run command will be
added here once the container has been tested and published. Until then, use
the Conda installation above and DIS optical flow on the CPU if you do not
already have CUDA-enabled OpenCV.

Check the installation
----------------------

Confirm that Python can import the main RodeOFlow functions:

.. code-block:: bash

   python -c "from rodeoflow import pose_face, pose_body, analyse_face, analyse_body; print('RodeOFlow import successful')"

This confirms that the main functions and their import dependencies are
available. It does not confirm that OpenCV can use CUDA.