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
├── configs/
│   └── Environment-specific configuration files
│
├── train.py
│   └── Training script for PPO variants (with or without SAPPS)
│
├── evaluate.py
│   └── Evaluation and plotting utilities
│
├── utils/
│   └── Helper functions and wrappers
│
└── README.md
```

*(Contents may vary slightly by task.)*

---

## Installation

Ensure that MuJoCo and Gymnasium are installed and configured.

```bash
pip install -r requirements.txt
```

A working MuJoCo installation (e.g., MuJoCo 2.x) is required. Refer to the official MuJoCo and Gymnasium documentation for platform-specific installation details.

---

## Running Experiments

To train a policy using SAPPS on a MuJoCo task (e.g., Walker2D):

```bash
python train.py --env Walker2D --method sapps
```

Baseline methods can be selected by changing the `--method` argument (e.g., `vanilla`, `caps`, `lipsnet`).

The `evaluate.py` script can be used to evaluate trained models and compute performance and smoothness metrics across multiple random seeds.

---

## Reproducibility

- All experiments are run with multiple random seeds.
- Results are reported as averages with variability measures (e.g., standard deviation).
- Due to stochastic initialization and simulation variability, exact results may differ slightly between runs.

---

## Relation to the Paper

The MuJoCo experiments serve as controlled benchmarks that isolate the effect of SAPPS on policy smoothness and performance, independent of domain-specific complexities (such as optical system dynamics or hardware constraints).

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

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and by the **National Research Council Canada (NRC)**. These experiments build upon the MuJoCo physics engine and the OpenAI Gymnasium framework.
