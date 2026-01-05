# SAPPS-RL
## Adaptive Policy Regularization for Smooth Control in Reinforcement Learning

This repository contains the official implementation of State-Adaptive Proportional Policy Smoothing (SAPPS), a policy regularization method designed to produce smooth yet responsive control policies in continuous-control reinforcement learning.

SAPPS suppresses high-frequency oscillations in learned policies without compromising performance, particularly in dynamic environments where rapid adaptation is required.

**Paper**: *Adaptive Policy Regularization for Smooth Control in Reinforcement Learning*  
**Status**: Under review  
**Authors**: Payam Parvizi, Abhishek Naik, Colin Bellinger, Ross Cheriton, Davide Spinello  
**Repository**: https://github.com/payamparvizi/sapps-rl  

---

## Abstract

A significant challenge in applying reinforcement learning (RL) to continuous-control problems is the presence of high-frequency oscillations in the actions produced by learned policies. These oscillations result in abrupt control responses, leading to excessive actuator wear, increased power consumption, and instability in real-world deployments. Existing approaches to reduce such oscillations often involve trade-offs, including increased architectural complexity or degraded policy performance, particularly in environments where rapid state changes require rapid adaptation.

To address this issue, we propose State-Adaptive Proportional Policy Smoothing (SAPPS), a novel approach that adaptively adjusts smoothness constraints to suppress high-frequency components in RL policies. SAPPS is inspired by Lipschitz continuity. It introduces a state-adaptive proportional regularization during policy optimization, encouraging changes in consecutive actions to scale with changes in consecutive observations. This adaptive constraint enables smooth yet responsive control.

Results from simulation and hardware experiments demonstrate that SAPPS produces smooth control policies without compromising performance across a diverse set of environments, including MuJoCo continuous-control benchmarks, a simulated adaptive optics system for optical satellite communications, and a real-world nano quadcopter, under both slowly and rapidly changing conditions.

---

## Method Overview

SAPPS is a general policy regularization technique that can be integrated into deep RL algorithms to improve policy smoothness in both static and dynamic continuous-control settings. Rather than directly penalizing action magnitude or velocity, SAPPS regularizes the change between consecutive actions based on the relative change between consecutive observations.

This adaptive formulation:
- penalizes unnecessary action fluctuations when state changes are small
- preserves responsiveness when large observation changes require rapid control adaptation

SAPPS is implemented within **Proximal Policy Optimization (PPO)** and compared against:
- Vanilla PPO  
- PPO with Conditioning for Action Policy Smoothness (CAPS)  
- PPO with LipsNet-based architectures  

---

## Repository Structure

```
sapps-rl/
├── Adaptive_Optics_Environment/
│   └── Wavefront sensorless adaptive optics environment
│
├── MuJoCo_Environments/
│   └── Standard MuJoCo continuous-control benchmark tasks
│
├── Quadcopter_Environment/
│   └── Real-world nano quadcopter hovering experiments
│
└── README.md
```

Each environment directory is self-contained and includes training and evaluation scripts corresponding to the experiments reported in the paper.

---

## Experimental Domains

### 1. MuJoCo Continuous-Control Benchmarks
SAPPS is evaluated on standard OpenAI Gymnasium MuJoCo tasks, including:
- Walker2D  
- HalfCheetah  
- Ant  
- Reacher  
- Swimmer  

Across these benchmarks, SAPPS improves policy smoothness while maintaining or improving task return.

### 2. Real-World Quadcopter Control
A nano quadcopter hovering task is used to validate real-world applicability. SAPPS demonstrates reduced control oscillations, improved actuator efficiency, and stable performance under sensor noise and external disturbances.

### 3. Wavefront Sensorless Adaptive Optics
A highly dynamic optical control problem inspired by satellite-to-ground optical communication. SAPPS is evaluated under quasi-static atmospheric conditions and rapidly changing turbulence (high drift velocities), maintaining performance where fixed smoothing methods significantly degrade.

---

## Installation

This repository assumes a standard Python-based reinforcement learning stack.

### Requirements
- Python ≥ 3.8  
- PyTorch  
- NumPy  
- OpenAI Gymnasium (with MuJoCo support)

Each environment subdirectory includes its own requirements.txt listing any additional dependencies (e.g., specialized simulation libraries or hardware interface packages).

---

## Running Experiments

Each environment directory contains its own training and evaluation scripts. Please refer to environment-specific scripts and configuration files for details on hyperparameters and experimental settings.

---

## Reproducibility

All reported results are averaged over multiple random seeds. Hyperparameter ranges match those described in the paper, and evaluation protocols are consistent across all compared methods. Due to simulator versions, hardware differences, and stochasticity, exact numerical reproduction may vary; however, qualitative trends are robust.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{parvizi2026sapps,
  title={Adaptive Policy Regularization for Smooth Control in Reinforcement Learning},
  author={Parvizi, Payam and Naik, Abhishek and Bellinger, Colin and Cheriton, Ross and Spinello, Davide},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2026},
  note={under review}
}
```

A `CITATION.cff` file will be added upon publication.

---

## License

This repository will be released under an open-source license upon publication.

---

## Contact

**Payam Parvizi**  
Email: pparv056@uottawa.ca  
GitHub: https://github.com/payamparvizi

---

## Acknowledgments

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and in part by the **National Research Council Canada (NRC)**.
