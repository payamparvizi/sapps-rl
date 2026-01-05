# Quadcopter Control Environment

This directory contains the **real-world quadcopter control environment** used to evaluate **State-Adaptive Proportional Policy Smoothing (SAPPS)** in the paper:

**Adaptive Policy Regularization for Smooth Control in Reinforcement Learning**

The experiments demonstrate the applicability of SAPPS to **physical robotic systems**, highlighting its ability to produce smooth and stable control signals under real-world noise and disturbances.

---

## Overview

The quadcopter experiments are designed to validate SAPPS beyond simulation by deploying learned policies on a **nano quadcopter** platform. Unlike simulated environments, real-world control introduces sensor noise, actuator delays, and unmodeled dynamics, making policy smoothness critical for stable operation.

This environment formulates the quadcopter hovering task as a **continuous-control Markov Decision Process (MDP)** and enables the evaluation of policy smoothness and performance on real hardware.

---

## Key Characteristics

- **Platform**  
  Nano quadcopter equipped with onboard sensors for attitude and altitude estimation.

- **Observation Space**  
  Low-dimensional state representation including position, velocity, and attitude-related measurements.

- **Action Space**  
  Continuous motor command signals controlling thrust and stabilization.

- **Disturbances**  
  Sensor noise, actuator delays, and external perturbations inherent to physical systems.

- **Reward Function**  
  Designed to encourage stable hovering while penalizing excessive control oscillations.

This setup emphasizes **actuator efficiency**, **control smoothness**, and **robustness**.

---

## Directory Structure

```
Quadcopter_Environment/
├── configs/
│   └── Experiment and controller configuration files
│
├── train.py
│   └── Training script for PPO-based controllers
│
├── deploy.py
│   └── Deployment script for running trained policies on hardware
│
├── utils/
│   └── Helper functions for logging and evaluation
│
└── README.md
```

*(Directory contents may vary depending on hardware configuration.)*

---

## Installation

This environment requires both software dependencies and compatible quadcopter hardware.

```bash
pip install -r requirements.txt
```

Additional setup may be required for communication with the quadcopter (e.g., radio drivers or firmware tools). Please refer to the hardware documentation for platform-specific instructions.

---

## Running Experiments

### Training (Simulation or Hardware-in-the-Loop)
```bash
python train.py
```

### Deployment on Hardware
```bash
python deploy.py
```

Training and deployment configurations are specified via configuration files and script arguments.

---

## Usage Notes

- These experiments involve **real hardware** and should be conducted with appropriate safety precautions.
- Policy behavior may vary across hardware units due to manufacturing tolerances and battery conditions.
- Exact numerical reproducibility is not expected; emphasis is placed on qualitative stability and smoothness improvements.

---

## Relation to the Paper

The quadcopter experiments provide a **hardware validation** of SAPPS, demonstrating that the proposed method improves policy smoothness and control stability in real-world robotic systems.

---

## Citation

If you use this code or experimental setup, please cite the associated paper:

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
