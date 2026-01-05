# Wavefront Sensorless Adaptive Optics Environment

This directory contains the **reinforcement learning (RL) environment for wavefront sensorless adaptive optics (AO)** used in the paper:

**Adaptive Policy Regularization for Smooth Control in Reinforcement Learning**

The environment is designed to evaluate **policy smoothness and control responsiveness** in **highly dynamic optical systems**, with a primary motivation drawn from **satellite-to-ground optical communication** scenarios.

---

## Overview

Wavefront sensorless adaptive optics aims to correct atmospheric wavefront distortions **without explicit wavefront sensing**, relying instead on low-dimensional photodetector measurements and closed-loop control of a deformable mirror.

This environment formulates the AO control problem as a **continuous-control Markov Decision Process (MDP)** and is used to evaluate reinforcement learning algorithms under conditions where:

- observations are low-dimensional and partially informative,
- dynamics can change rapidly due to atmospheric turbulence, and
- excessive control oscillations directly degrade optical performance.

The focus of this environment within the current repository is to assess whether **SAPPS** enables **smooth yet responsive control** in such challenging regimes.

---

## Relation to Prior Work

This environment builds conceptually on prior work in reinforcement learning for wavefront sensorless adaptive optics, including earlier AO-RL simulation frameworks developed by the authors. In particular, related background and environment design details can be found in:

**Parvizi et al., _Reinforcement Learning Environment for Wavefront Sensorless Adaptive Optics in Single-Mode Fiber Coupled Optical Satellite Communications Downlinks_, Photonics (2023)**  
🔗 https://doi.org/10.3390/photonics10121371

However:

- the **learning objectives**,  
- the **control regularization strategy**, and  
- the **experimental focus on action smoothness under dynamic conditions**  

in the present work are distinct from those in the above publication.

Importantly, this repository does **not** aim to reproduce or extend the Photonics (2023) results. Instead, it reuses a compatible simulation framework to provide a **controlled and challenging testbed** for evaluating adaptive policy regularization.

---

## Environment Characteristics

- **Observation Space**  
  Low-dimensional photodetector measurements derived from the focal plane intensity distribution (e.g., quadrant or coarse pixel arrays).

- **Action Space**  
  Continuous control commands applied to a deformable mirror, parameterized either directly in actuator space or via low-order modal representations.

- **Dynamics**  
  Atmospheric turbulence modeled using phase screens with configurable strength and drift velocity, enabling quasi-static to rapidly varying regimes.

- **Reward Function**  
  Optical performance metrics related to coupling efficiency or Strehl-like measures.

This design emphasizes **low latency**, **partial observability**, and **fast temporal dynamics**, making it particularly sensitive to high-frequency control oscillations.

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

Some dependencies (e.g., optical simulation libraries) may require system-level packages. Please refer to the comments in `requirements.txt` if installation issues arise.

---

## Running the Environment

To train or evaluate an RL agent in the wavefront sensorless AO environment:

```bash
python run_wslao.py
```

Key configuration parameters—such as atmospheric regime, turbulence velocity, observation dimension, and action parameterization—are defined within the script and associated configuration files.

---

## Usage Notes

- This environment is intended for **research and comparative evaluation purposes**.
- Exact numerical reproducibility is not expected due to stochastic turbulence generation.
- Emphasis is placed on **relative performance and smoothness comparisons** between methods rather than absolute optical metrics.

---

## Relation to the Paper

The adaptive optics experiments reported in the paper use this environment to demonstrate that **state-adaptive policy regularization** improves control smoothness and robustness in highly dynamic, partially observed systems.

The environment is **not required** to apply SAPPS in other domains such as MuJoCo benchmarks or robotic control tasks.

---

## Citation

If you use this environment or build upon it, please cite the associated paper:

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

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and the **National Research Council Canada (NRC)**.

