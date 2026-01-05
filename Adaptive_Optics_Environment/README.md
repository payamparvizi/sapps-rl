# Wavefront Sensorless Adaptive Optics Environment

This directory contains the **reinforcement learning (RL) environment for wavefront sensorless adaptive optics (AO)** used in the paper:

*Adaptive Policy Regularization for Smooth Control in Reinforcement Learning*

The environment is designed to evaluate **policy smoothness and control responsiveness** in **highly dynamic optical systems**, with a primary motivation drawn from **satellite-to-ground optical communication** scenarios.

---

## Overview

Wavefront sensorless adaptive optics aims to correct atmospheric wavefront distortions **without explicit wavefront sensing**, relying instead on low-dimensional photodetector measurements and closed-loop control of a deformable mirror.

This environment formulates the AO control problem as a **continuous-control Markov Decision Process (MDP)** and is used to evaluate reinforcement learning algorithms under conditions where:

- observations are low-dimensional and partially observable,
- dynamics can change rapidly due to atmospheric turbulence, and
- excessive control oscillations directly degrade optical performance.

The focus of this environment is to assess whether **SAPPS** enables **smooth yet responsive control under** such challenging conditions.

---

## Relation to Prior Work

This environment builds on prior research in RL for wavefront sensorless adaptive optics, including an earlier AO-RL simulation framework by the authors (see **Parvizi et al., *Reinforcement Learning Environment for Wavefront Sensorless Adaptive Optics in Single-Mode Fiber Coupled Optical Satellite Communications Downlinks*, Photonics 2023** – [https://doi.org/10.3390/photonics10121371](https://doi.org/10.3390/photonics10121371))

However, the present work differs in key aspects:

- **Learning objectives** – new objectives tailored to adaptive regularization
- **Control regularization strategy** – introduces the SAPPS method instead of fixed smoothness penalties
- **Experimental focus** – emphasis on action smoothness under dynamic conditions

Importantly, this repository does *not* aim to reproduce or extend the Photonics 2023 results. Instead, it reuses a compatible simulation framework as a **controlled and challenging testbed** for evaluating adaptive policy regularization methods.

---

## Environment Characteristics

- **Observation Space**  
  Low-dimensional photodetector measurements derived from the focal-plane intensity distribution (e.g., quadrant or coarse pixel array).

- **Action Space**  
  Continuous control commands applied to a deformable mirror, parameterized either directly in actuator space or via low-order modal representations.

- **Dynamics**  
  Atmospheric turbulence modeled using phase screens with configurable strength and drift velocity, spanning quasi-static to rapidly varying regimes.

- **Reward Function**  
  Optical performance metrics related to coupling efficiency or a Strehl ratio proxy.

This design emphasizes **low latency, partial observability**, and **fast temporal dynamics**, making it particularly sensitive to high-frequency control oscillations.

---

## Directory Structure

```
Adaptive_Optics_Environment/
├── gym_AO/
│   └── Gymnasium-compatible AO environment
│
├── packages/
│   └── Optical modeling and simulation utilities
│
├── utils/
│   └── Helper functions for environment simulation and evaluation
│
├── run_wslao.py
│   └── Entry point for training and evaluation
│
├── requirements.txt
│   └── Environment-specific Python dependencies
│
└── README.md
```

---

## Installation

It is recommended to use a dedicated Python environment (e.g., `venv` or Conda) for this module.

```bash
pip install -r requirements.txt
```

Some dependencies (e.g., optical simulation libraries) may require system-level packages. Please consult the comments in `requirements.txt` if any installation issues arise.

---

## Running the Environment

To train or evaluate an RL agent in the wavefront sensorless AO environment:

```bash
python run_wslao.py
```

Key configuration parameters—such as atmospheric conditions, turbulence drift velocity, observation dimensions, and action parameterization—are defined within the script and associated configuration files.

---

## Reproducibility

- This environment is intended for **research and comparative evaluation** purposes.
- Exact numerical reproducibility is not expected due to stochastic turbulence generation.
- Emphasis is on **relative performance** and smoothness comparisons rather than absolute optical metrics.

---

## Relation to the Paper

The adaptive optics experiments in the paper use this environment to show that **state-adaptive policy regularization (SAPPS)** improves control smoothness and robustness in a highly dynamic, partially observed system.

*Note:* This environment is **not required** to apply SAPPS in other domains (e.g., MuJoCo or robotic control).

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

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and by the **National Research Council Canada (NRC)**.


