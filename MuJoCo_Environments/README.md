# MuJoCo Continuous-Control Environments

This directory contains the **MuJoCo continuous-control benchmark environments** used to evaluate **State-Adaptive Proportional Policy Smoothing (SAPPS)** in the paper:

*Adaptive Policy Regularization for Smooth Control in Reinforcement Learning*

These tasks serve as standardized benchmarks to assess policy smoothness and overall performance in simulated robotic control scenarios.

---

## Overview

MuJoCo environments are widely used for evaluating reinforcement learning algorithms in continuous control settings. They provide well-understood dynamics and standardized evaluation protocols, making them ideal for isolating the effects of a policy regularization method.

In these benchmarks, SAPPS:
- improves policy smoothness
- suppresses high-frequency action oscillations
- maintains or improves task performance relative to baseline methods

---

## Environments Included

The following OpenAI Gymnasium MuJoCo tasks are used:

- **Walker2D**
- **HalfCheetah**
- **Ant**
- **Reacher**
- **Swimmer**

All environments have continuous state and action spaces, and all experiments use identical training and evaluation protocols for fair comparison.

---

## Experimental Setup

- **RL Algorithm**: Proximal Policy Optimization (PPO)
- **Policy Types Compared**:
  - Vanilla PPO
  - CAPS (Conditioning for Action Policy Smoothness)
    
- **Evaluation Metrics**:
  - Average episodic return
  - Policy smoothness (action fluctuation)

Hyperparameter ranges and evaluation procedures match those reported in the paper.

---

## Directory Structure

```
MuJoCo_Environments/
├── packages/
│ └── Local copy of the Tianshou reinforcement learning framework
│
├── utils/
│   ├── arguments/
│   │   └── Experiment configuration and command-line argument definitions
│   │
│   └── compute_smoothness.py
│       └── Utility for computing **volatility measure** in MuJoCo environments
│
├── run_mujoco.py
│ └── Entry point for training and evaluating PPO-based methods
│
├── requirements.txt
│ └── Python dependencies for MuJoCo experiments
│
└── README.md
```

---

## Installation

Ensure that MuJoCo and Gymnasium are installed and configured.

```bash
pip install -r requirements.txt
```

A working MuJoCo installation (e.g., MuJoCo 2.1) is required. Refer to the official MuJoCo and Gymnasium documentation for platform-specific installation details.

---

## Running the Environment

To train or evaluate an RL agent on the MuJoCo continuous-control benchmarks, run:

```bash
python run_mujoco.py
```

### Selecting the MuJoCo environment

MuJoCo-specific experiment configurations are defined in:

```
utils/arguments/
├── arguments.py
├── arguments_walker2d.py
├── arguments_halfcheetah.py
├── arguments_ant.py
├── arguments_reacher.py
└── arguments_swimmer.py
```

Each `arguments_<env>.py` file defines the full experimental configuration for a specific MuJoCo task, including environment settings and training hyperparameters.

By default, `run_mujoco.py` loads its configuration via:

```
from utils.arguments.arguments import get_args
```

This default configuration corresponds to the Ant environment, as used in the main MuJoCo evaluations reported in the paper.

To run experiments on a different MuJoCo task, select the appropriate preset by modifying the configuration selection inside `utils/arguments/arguments.py`, or by directly importing the desired preset module, for example:
- `arguments_walker2d.py` — Walker2D-v4
- `arguments_halfcheetah.py` — HalfCheetah-v4
- `arguments_ant.py` — Ant-v4
- `arguments_reacher.py` — Reacher-v4
- `arguments_swimmer.py` — Swimmer-v4

### Selecting the Policy Regularization Method

The MuJoCo experiments support multiple policy regularization methods through a command-line argument. The selected option determines the **policy regularization strategy** used during training.

#### Policy selection argument

The regularization method is selected using:

```bash
--regularization_case
```

Available options are:
- `standard_PPO`: Vanilla PPO without policy smoothing
- `PPO_CAPS`: PPO with **Conditioning for Action Policy Smoothness (CAPS)**
- `PPO_SAPPS`: PPO with **State-Adaptive Proportional Policy Smoothing (SAPPS)** (proposed method)

By default:
```bash
--regularization_case standard_PPO
```

All methods are evaluated using the same network architecture and training protocol, ensuring that performance differences are attributable to the regularization strategy rather than architectural changes.

---

## Relation to the Paper

The MuJoCo experiments provide standardized continuous-control benchmarks used to quantitatively compare SAPPS against baseline policy regularization methods in terms of both task performance and action smoothness.

---

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{parvizi2026sapps,
  title={Adaptive Policy Regularization for Smooth Control in Reinforcement Learning},
  author={Parvizi, Payam and Naik, Abhishek and Bellinger, Colin and Cheriton, Ross and Spinello, Davide},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2026},
  note={under review}
}
```

---

## Acknowledgments

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and by the **National Research Council Canada (NRC)**. These experiments build upon the MuJoCo physics engine and the Farama Gymnasium framework.
