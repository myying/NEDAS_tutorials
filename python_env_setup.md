# Setup native python environment for the tutorials

## Clone the NEDAS_tutorial repositories

``git clone https://github.com/myying/NEDAS_tutorials.git``

## Create python env

Note: Python > 3.12 is required for NEDAS.

### Use python virtual env

Create the environment: 
``python -m venv ~/nedas`` where ``~/nedas`` is its location.

Enter the environment:
``. ~/nedas/bin/activate``

### Use conda

Create the environment:
``conda create -n nedas python=3.12``

Enter the environment:
``conda activate nedas``

## Install NEDAS

### From PyPI

NEDAS is available from the PyPI platform. You can install with ``pip install NEDAS``

### Manual install in editable mode

You can also manually clone the repository 
``git clone https://github.com/nansencenter/NEDAS.git``

and install in editable mode
``cd NEDAS; pip install -e .`` for active code development

## Install dependencies

For the tutorials, additional dependencies are required:
``pip install numba cmocean jupyter ipywidgets``

Note: the notebook can be run in single processor mode (``nproc=1``),
but the Docker image comes with MPICH support, you may install the ``mpi4py`` module to run it with ``nproc>1``. To install ``mpi4py`` on your native python environment, you need the MPICH (not OpenMP) and ``pip install mpi4py``.

## Launch jupyter lab

Inside NEDAS_tutorials directory (where you've cloned it), start the server ``jupyter lab``
