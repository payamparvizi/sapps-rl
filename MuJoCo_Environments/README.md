# MuJoCo Continuous-Control Environments

This directory contains the **MuJoCo continuous-control benchmark environments** used to evaluate the effectiveness of **State-Adaptive Proportional Policy Smoothing (SAPPS)** in the paper:

**Adaptive Policy Regularization for Smooth Control in Reinforcement Learning**

These environments serve as standardized benchmarks to assess policy smoothness and task performance in simulated robotic control tasks.

---

## Overview

MuJoCo environments are widely used benchmarks for evaluating reinforcement learning algorithms in continuous-control settings. They provide well-understood dynamics and standardized evaluation protocols, making them suitable for isolating the effects of policy regularization methods.

In this work, MuJoCo tasks are used to demonstrate that SAPPS:
- improves policy smoothness,
- suppresses high-frequency action oscillations, and
- maintains or improves task performance relative to baseline methods.

---

## Environments Included

The following OpenAI Gymnasium MuJoCo environments are used:

- **Walker2D**
- **HalfCheetah**
- **Ant**
- **Reacher**
- **Swimmer**

All environments involve continuous state and action spaces and are evaluated under identical training and evaluation protocols for fair comparison.

---

## Experimental Setup

- **RL Algorithm**: Proximal Policy Optimization (PPO)  
- **Policy Types Compared**:
  - Vanilla PPO
  - PPO + CAPS (Conditioning for Action Policy Smoothness)
  - PPO + LipsNet
  - PPO + SAPPS
- **Evaluation Metrics**:
  - Average episodic return
  - Policy smoothness metrics based on action variation

Hyperparameter ranges and evaluation protocols match those reported in the paper.

---

## Directory Structure

```
MuJoCo_Environments/
├── configs/
│   └── Environment-specific configuration files
│
├── train.py
│   └── Training script for PPO-based methods
│
├── evaluate.py
│   └── Evaluation and plotting utilities
│
├── utils/
│   └── Helper functions and wrappers
│
└── README.md
```

*(Directory contents may vary slightly depending on the environment.)*

---

## Installation

These environments require MuJoCo and Gymnasium support.

```bash
pip install -r requirements.txt
```

You must also have a working MuJoCo installation compatible with your operating system. Please refer to the official MuJoCo and Gymnasium documentation for installation instructions.

---

## Running Experiments

To train a policy using SAPPS on a MuJoCo environment:

```bash
python train.py --env Walker2D --method sapps
```

Baseline methods can be selected by changing the `--method` argument.

Evaluation scripts are provided to compute performance and smoothness metrics across multiple random seeds.

---

## Usage Notes

- All experiments are run with multiple random seeds.
- Results are reported as averages with variability measures.
- Exact numerical results may vary due to stochastic initialization and simulator differences.

---

## Relation to the Paper

The MuJoCo experiments provide controlled benchmarks that isolate the effect of SAPPS on policy smoothness and performance, independent of domain-specific complexities such as optical systems or hardware constraints.

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

These experiments build upon the MuJoCo physics engine and the OpenAI Gymnasium benchmarking framework.
