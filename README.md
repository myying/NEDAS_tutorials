# Jupyter notebook tutorials for NEDAS

Author: [Yue Ying](https://myying.github.io/) (NERSC)

Data assimilation (DA) combines information from model forecasts and observations to obtain the best estimate of model state and parameters.
NEDAS provides a light-weight solution for developing new DA algorithms for Earth-system models. 

In this series of jupyter notebook tutorials, I demonstrate how to use NEDAS to perform DA research.

To run the notebooks, you can use one of these options:
[EDITO Datalab](https://datalab.dive.edito.eu/launcher/ocean-modelling/nedas-tutorials?name=nedas-tutorials&version=1.3.0&autoLaunch=true),
[Google Colab](https://colab.research.google.com/github/myying/NEDAS_tutorials/),
[Run in Docker](https://hub.docker.com/r/myying/nedas-tutorials),
or [Run in native environment](https://github.com/myying/NEDAS_tutorials/blob/main/python_env_setup.md).

A summary of purpose of each notebook:
- [1.step_by_step_with_vort2d_case.ipynb](https://github.com/myying/NEDAS_tutorials/blob/main/1.step_by_step_with_vort2d_case.ipynb) provides a step-by-step guide to running DA experiments using a simple 2D vorticity model as example.
- [2.validation_with_lorenz96.ipynb](https://github.com/myying/NEDAS_tutorials/blob/main/2.validation_with_lorenz96.ipynb) validates the NEDAS for correctness of implementation, using the classic Lorenz 1996 model cases.

Learn more from the [NEDAS Documentation](https://nedas.readthedocs.io/en/latest/)
