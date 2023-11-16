<div align="center">
    <h2><b style="color:red;">Warning: this is a pre-release, API is subject to major changes</b></h2>
</div>

<div align="center"><img alt="dm_robotics_panda Logo" src="https://raw.githubusercontent.com/JeanElsner/dm_robotics_panda/main/logo.png?token=GHSAT0AAAAAAB4JT3DTYA4QRPHPLIIDLETWZKT2XDQ" /></div>

<h2 align="center">Model and Tools for the Panda Robot in dm_robotics</h2>

<p align="center">
<a href="https://github.com/JeanElsner/dm_robotics_panda/actions/workflows/main.yml"><img alt="Continuous Integration" src="https://img.shields.io/github/actions/workflow/status/JeanElsner/dm_robotics_panda/main.yml" /></a>
<a href="https://github.com/JeanElsner/dm_robotics_panda/blob/main/LICENSE"><img alt="Apache-2.0 License" src="https://img.shields.io/github/license/JeanElsner/dm_robotics_panda" /></a>
<a href="https://jeanelsner.github.io/dm_robotics_panda/pylint.log"><img alt="Pylint score" src="https://jeanelsner.github.io/dm_robotics_panda/pylint.svg" /></a>
<a href="https://pypi.org/project/dm-robotics-panda/"><img alt="PyPI - Published Version" src="https://img.shields.io/pypi/v/dm-robotics-panda"></a>
<a href="https://codecov.io/gh/JeanElsner/dm_robotics_panda"><img src="https://codecov.io/gh/JeanElsner/dm_robotics_panda/graph/badge.svg?token=7mk9f5yM8y"/></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dm-robotics-panda">
</p>

This package includes a model of the Panda robot for [dm_robotics](https://github.com/google-deepmind/dm_robotics), tools to build simulation environments suited for reinforcement learning, and allows you to run these simulations with real hardware in the loop. 

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
Hardware-in-the-loop operation requires `panda-py` to control the robot. However, the version automatically installed from PyPI may not be compatible with your robot if you use an older firmware or use the new Franka Research 3. In that case refer to the panda-py [instructions](https://github.com/JeanElsner/panda-py#libfranka-version) on what version to install and where to find it.

For visualization, additional libraries are required, depending on whether hardware acceleration and/or headless mode is required. On Ubuntu 22.04 you can install the respective requirements by running
* Hardware accelerated rendering in windowed mode `sudo apt-get install libglfw3 libglew2.2`
* Headless hardware acceleration with recent NVIDIA driver `sudo apt-get install libglew2.2`
* Software rendering `sudo apt-get install libgl1-mesa-glx libosmesa6`

Additionally you may need to install glib `sudo apt-get install libglib2.0-0`.
