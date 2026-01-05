# Quadcopter Control Environment (Crazyflie 2.1)

This directory contains the **real-world quadcopter control environment** used to evaluate **State-Adaptive Proportional Policy Smoothing (SAPPS)** in the paper:

**Adaptive Policy Regularization for Smooth Control in Reinforcement Learning**

The experiments are conducted on a **Bitcraze Crazyflie 2.1 nano quadcopter** and demonstrate the effectiveness of SAPPS in producing **smooth, stable, and hardware-safe control policies** under real-world noise, delays, and disturbances.

---

## Overview

While simulation benchmarks are useful for controlled evaluation, real-world robotic systems introduce additional challenges, including sensor noise, actuator delays, communication latency, and unmodeled dynamics. These effects can significantly amplify high-frequency oscillations in learned policies.

This environment formulates quadcopter hovering as a **continuous-control Markov Decision Process (MDP)** and enables training and deployment of reinforcement learning policies on physical hardware. It is specifically designed to evaluate whether SAPPS improves:

- control smoothness,
- actuator efficiency, and
- stability during real-world flight.

---

## Hardware Platform

- **Quadcopter**: Bitcraze Crazyflie 2.1  
- **Radio**: Crazyradio PA  
- **Sensor Deck**: Flow Deck v2  
- **Task**: Altitude stabilization (hovering)

The Crazyflie platform is selected due to its widespread use in research, lightweight design, and sensitivity to control oscillations.

---

## Environment Characteristics

- **Observation Space**  
  Low-dimensional state representation including altitude, vertical velocity, and attitude-related measurements.

- **Action Space**  
  Continuous control commands corresponding to motor thrust adjustments.

- **Disturbances**  
  Sensor noise, actuator delays, communication latency, and external perturbations.

- **Reward Function**  
  Encourages stable hovering while penalizing excessive action fluctuations.

This setup emphasizes **smooth control**, **robustness**, and **hardware safety**.

---

## Directory Structure

```
Quadcopter_Environment/
├── utils/
│   └── Argument parsing, logging, and helper utilities
│
├── ppo/
│   └── PPO agent and neural network definitions
│
├── crazyflie_env/
│   └── Environment definition (simulation and real hardware)
│
├── train.py
│   └── Main training script
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
```

*(Directory contents may vary slightly depending on configuration.)*

---

## Installation

### System Requirements
- Linux operating system
- Python ≥ 3.9
- Crazyflie 2.1 hardware and Crazyradio PA

It is recommended to use a virtual environment (e.g., Conda or venv).

```bash
pip install -r requirements.txt
```

The environment relies on the **Bitcraze Crazyflie Python library (cflib)**. Please ensure that Crazyflie drivers and dependencies are correctly installed.

---

## Hardware Setup and Verification

Before running RL experiments, verify communication with the Crazyflie:

```bash
cfclient
```

Ensure that:
- the Crazyradio is detected,
- the Crazyflie connects successfully, and
- onboard sensor data updates correctly.

Flow Deck attachment should be verified before flight. Refer to the Bitcraze documentation for detailed setup and troubleshooting instructions.

---

## Running Experiments

### Training in Simulation
```bash
python train.py --task simulation
```

### Training on Real Hardware
```bash
python train.py --task real
```

### Example: Setting Target Altitude and Action Limits
```bash
python train.py --task real --target_altitude 1.0 --action_range 0.20
```

Configuration parameters can also be modified in the argument definitions under `utils/`.

---

## Safety Notes

- These experiments involve **real flying hardware**.
- Always operate in a clear, enclosed space.
- Use conservative action limits during initial testing.
- Never deploy untested policies directly on hardware.

---

## Relation to the Paper

The quadcopter experiments provide **hardware validation** of SAPPS, demonstrating that the proposed method improves policy smoothness and control stability beyond simulation benchmarks.

---

## Citation

If you use this environment or experimental setup, please cite:

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
The implementation builds upon the Bitcraze Crazyflie ecosystem and associated open-source tools.
