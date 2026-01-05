# Wavefront Sensorless Adaptive Optics Environment

This directory contains the **reinforcement learning (RL) environment for wavefront sensorless adaptive optics (AO)** used in the paper:

**Adaptive Policy Regularization for Smooth Control in Reinforcement Learning**

The environment is designed to evaluate policy smoothness and control performance in **highly dynamic optical systems**, motivated by **satellite-to-ground optical communication**.

---

## Overview

Wavefront sensorless adaptive optics aims to correct atmospheric wavefront distortions **without explicit wavefront sensing**, relying instead on low-dimensional photodetector measurements and closed-loop control.

This environment formulates the AO control problem as a **continuous-control Markov Decision Process (MDP)** and enables the evaluation of deep RL algorithms under:

- quasi-static atmospheric turbulence  
- rapidly changing turbulence with high drift velocities  

The environment is used to assess the ability of SAPPS and baseline methods to produce **smooth yet responsive control signals** in highly dynamic conditions.

---

## Key Characteristics

- **Observation Space**  
  Low-dimensional photodetector measurements derived from the focal plane intensity distribution.

- **Action Space**  
  Continuous control commands applied to a deformable mirror model.

- **Dynamics**  
  Atmospheric turbulence modeled using phase screens with configurable drift velocity and strength.

- **Reward Function**  
  Based on optical performance metrics (e.g., fiber coupling efficiency / Strehl-related measures).

This design prioritizes **low latency**, **partial observability**, and **fast dynamics**, making it a challenging benchmark for RL-based control.

---

## Directory Structure

```
Adaptive_Optics_Environment/
├── gym_AO/
│   └── Custom Gymnasium-compatible AO environment
│
├── packages/
│   └── Optical modeling and environment dependencies
│
├── utils/
│   └── Helper functions for simulation and evaluation
│
├── run_wslao.py
│   └── Entry point for training and evaluation
│
├── requirements.txt
│   └── Environment-specific dependencies
│
└── README.md
```

---

## Installation

It is recommended to use a **separate virtual environment** for this module.

```bash
pip install -r requirements.txt
```

Some dependencies may require system-level libraries (e.g., for numerical or optical simulation). Please refer to the comments in `requirements.txt` if issues arise.

---

## Running the Environment

To train or evaluate an RL agent in the wavefront sensorless AO environment:

```bash
python run_wslao.py
```

Configuration options (e.g., turbulence strength, drift velocity, RL algorithm settings) are specified within the script and related configuration files.

---

## Usage Notes

- This environment is intended for **research and evaluation purposes**
- Exact numerical reproducibility may vary due to stochastic turbulence generation
- The implementation prioritizes **relative comparisons** between methods rather than absolute optical performance

---

## Relation to the Paper

This environment is used exclusively for the **adaptive optics experiments** reported in the paper.  
It is **not required** to understand or apply the SAPPS method in other domains such as MuJoCo or robotics.

---

## Citation

If you use this environment in your research, please cite the associated paper:

```bibtex
@article{parvizi2025sapps,
  title={Adaptive Policy Regularization for Smooth Control in Reinforcement Learning},
  author={Parvizi, Payam and Naik, Abhishek and Bellinger, Colin and Cheriton, Ross and Spinello, Davide},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2025},
  note={under review}
}
```

---

## Acknowledgments

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and the **National Research Council Canada (NRC)**.
