<div align="center"><img alt="dm_robotics_panda Logo" src="https://raw.githubusercontent.com/JeanElsner/dm_robotics_panda/main/.github/img/logo.png" /></div>

<h2 align="center">Model and Tools for the Panda Robot in dm_robotics</h2>

<p align="center">
  <a href="https://github.com/JeanElsner/dm_robotics_panda/actions/workflows/main.yml"><img alt="Continuous Integration" src="https://img.shields.io/github/actions/workflow/status/JeanElsner/dm_robotics_panda/main.yml" /></a>
  <a href="https://github.com/JeanElsner/dm_robotics_panda/blob/main/LICENSE"><img alt="Apache-2.0 License" src="https://img.shields.io/github/license/JeanElsner/dm_robotics_panda" /></a>
  <a href="https://jeanelsner.github.io/dm_robotics_panda/pylint.log"><img alt="Pylint score" src="https://jeanelsner.github.io/dm_robotics_panda/pylint.svg" /></a>
  <a href="https://pypi.org/project/dm-robotics-panda/"><img alt="PyPI - Published Version" src="https://img.shields.io/pypi/v/dm-robotics-panda"></a>
  <a href="https://codecov.io/gh/JeanElsner/dm_robotics_panda"><img src="https://codecov.io/gh/JeanElsner/dm_robotics_panda/graph/badge.svg?token=7mk9f5yM8y"/></a>
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dm-robotics-panda">
  <a href="https://jeanelsner.github.io/dm_robotics_panda"><img alt="Documentation" src="https://shields.io/badge/-Documentation-informational" /><a/>
</p>

This package includes a model of the Panda robot for [dm_robotics](https://github.com/google-deepmind/dm_robotics), tools to build simulation environments suited for reinforcement learning, and allows you to run these simulations with real hardware in the loop. 

<div align="center">
  <img alt="Hardware in the loop operation." src="https://raw.githubusercontent.com/JeanElsner/dm_robotics_panda/main/.github/img/hil_mode.gif" />
  <p>Run your <code>dm_robotics</code> simulation environment on the real hardware without modification.</p>
</div>

<div align="center">
  <img alt="Haptic interaction mode." src="https://raw.githubusercontent.com/JeanElsner/dm_robotics_panda/main/.github/img/haptic_mode.gif" />
  <p>Haptic interaction mode allows you to physically interact with the simulation environment.</p>
</div>

To get started, checkout the [tutorial](https://jeanelsner.github.io/dm_robotics_panda/tutorial.html).

## Install
The recommended way of installing is using PyPI:
```
pip install dm-robotics-panda
```
Alternatively, you can install this package from source by executing
```
pip install .
```
in the repository's root directory.
## Requirements
Hardware-in-the-loop operation requires `panda-py` to control the robot. However, the version automatically installed from PyPI may not be compatible with your robot if you use an older firmware or use the new Franka Research 3 robot. In that case refer to the panda-py [instructions](https://github.com/JeanElsner/panda-py#libfranka-version) on what version to install and where to find it.

If you're having trouble running the included viewer or rendering scenes, please refer to the requirements of [dm_control](https://github.com/google-deepmind/dm_control#rendering).

## Citation

If you use dm_robotics_panda in published research, please consider citing the [original software paper](https://www.sciencedirect.com/science/article/pii/S2352711023002285).

```
@article{elsner2023taming,
title = {Taming the Panda with Python: A powerful duo for seamless robotics programming and integration},
journal = {SoftwareX},
volume = {24},
pages = {101532},
year = {2023},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2023.101532},
url = {https://www.sciencedirect.com/science/article/pii/S2352711023002285},
author = {Jean Elsner}
}
```
