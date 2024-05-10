# mushr-mujoco-gym
gym environment for mushr car and block

# Installation
Install [gymnasium with mujoco](https://gymnasium.farama.org/environments/mujoco/) dependencies:
```bash
pip install gymnasium[mujoco]
```

Install [Mujoco](https://mujoco.org/)(>=3.0.0) from website. Python dependencies can be installed with:
```bash
pip install mujoco
```


Install this environment:
```bash
cd mushr-mujoco_gym
pip install -e .
```

# Errors

`attributeerror: 'mujoco._structs.mjdata' object has no attribute 'solver_iter'. did you mean: 'solver_niter'?`: \
Change `mujoco_rendering.py` as done in this [pull request](https://github.com/Farama-Foundation/Gymnasium/pull/746/files)


