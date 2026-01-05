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

However, the present work differs in its formulation and evaluation focus, introducing state-adaptive policy regularization (SAPPS) and explicitly analyzing action smoothness and control robustness under dynamic atmospheric conditions.

Importantly, this repository does *not* aim to reproduce or extend the Photonics 2023 results. Instead, it reuses a compatible simulation framework as a **controlled and challenging testbed** for evaluating adaptive policy regularization methods.

---

## Environment Characteristics

- **Observation Space**  
  Low-dimensional photodetector measurements derived from the focal-plane intensity distribution.

- **Action Space**  
  Continuous control commands applied to a deformable mirror, parameterized using low-order Zernike modal representations.

- **Dynamics**  
  Atmospheric turbulence modeled using phase screens with configurable strength and drift velocity, spanning slowly to rapidly varying regimes.

- **Reward Function**  
  Optical performance metrics related to coupling efficiency.

This design emphasizes **low latency, partial observability**, and **fast temporal dynamics**, making it particularly sensitive to high-frequency control oscillations.

---

## Directory Structure

```
Adaptive_Optics_Environment/
├── gym_AO/
│   └── Gymnasium-compatible AO environment
│
├── packages/
│   └── Auxiliary packages for reinforcement learning and baseline methods
│
├── utils/
│   └── Experiment configuration, network definitions, and supporting utilities
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

---

## Running the Environment

To train or evaluate an RL agent in the wavefront sensorless adaptive optics (WSL-AO) environment, run:

```bash
python run_wslao.py
```

### Selecting the atmospheric regime (turbulence drift velocity)

The WSL-AO environment uses **velocity-specific configuration files** located in:

```
utils/arguments/
├── arguments.py
├── arguments_v_5mps.py
├── arguments_v_50mps.py
└── arguments_v_500mps.py
```

Each `arguments_v_*` file defines the full experimental configuration for a specific atmospheric turbulence drift velocity, including environment parameters and training hyperparameters.

By default, `run_wslao.py` loads its configuration via:

```bash
from utils.arguments.arguments import get_args
```

This default configuration corresponds to 50 m/s atmospheric drift velocity, as used in the main experimental evaluations reported in the paper.

To evaluate SAPPS under different atmospheric dynamics, select the appropriate preset by modifying the configuration selection inside `utils/arguments/arguments.py`, or by directly importing the desired preset module:

- `arguments_v_5mps.py` — slowly varying atmospheric turbulence
- `arguments_v_50mps.py` — moderately fast atmospheric turbulence (default)
- `arguments_v_500mps.py` — highly dynamic atmospheric turbulence

#### Notes
- Logging and monitoring utilities (Weights & Biases) are initialized in run_wslao.py and can be enabled or disabled there as needed.
- For reproducibility and consistency with the paper, the provided velocity-specific presets should be used without modification when reproducing reported results.

### Selecting the Policy Regularization Method

.....


---

## Reproducibility

- This environment is intended for **research and comparative evaluation** purposes.
- Exact numerical reproducibility is not expected due to stochastic turbulence generation.
- Emphasis is on **relative performance** and smoothness comparisons rather than absolute optical metrics.

---

## Relation to the Paper

The adaptive optics experiments in the paper use this environment to show that **state-adaptive policy regularization (SAPPS)** improves control smoothness and robustness in highly dynamic environments.

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

















