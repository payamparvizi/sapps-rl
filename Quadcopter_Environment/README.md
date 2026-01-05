# Quadcopter Control Environment (Crazyflie 2.1)

This directory contains the **real-world quadcopter control environment** used to evaluate **State-Adaptive Proportional Policy Smoothing (SAPPS)** in the paper:

*Adaptive Policy Regularization for Smooth Control in Reinforcement Learning*

Experiments are conducted on a **Bitcraze Crazyflie 2.1** nano quadcopter and demonstrate the effectiveness of SAPPS in producing **smooth, stable, and hardware-safe** control policies under real-world noise, delays, and disturbances.

---

## Overview

While simulation benchmarks are useful for controlled evaluation, real-world robotic systems introduce additional challenges such as sensor noise, actuator delays, communication latency, and unmodeled dynamics. These effects can significantly amplify high-frequency oscillations in learned policies.

This environment supports both **simulation-based training** and **real-world deployment** on a Crazyflie quadcopter. Quadcopter hovering is formulated as a **continuous-control Markov Decision Process (MDP)**, enabling training and deployment of reinforcement learning policies on physical hardware. It is designed to evaluate SAPPS’s impact on:

- control smoothness,
- actuator efficiency, and
- stability during real-world flight.


---

## Hardware Platform

- **Quadcopter**: Bitcraze Crazyflie 2.1
- **Radio**: Crazyradio PA (2.4 GHz USB radio)
- **Sensor Deck**: Flow Deck v2 
- **Task**: Altitude stabilization (hovering)

The Crazyflie platform is chosen for its widespread use in research and lightweight design.

---

## Environment Characteristics

- **Observation Space**  
  A low-dimensional state representation consisting of the current altitude and vertical velocity, obtained either from simulated dynamics or from onboard state estimation during real-world operation

- **Action Space**  
  A single continuous control input corresponding to the commanded vertical velocity, bounded within a predefined action range to ensure safe operation.

- **Reward Function**  
  The reward is primarily based on the absolute altitude error relative to the target altitude.

- **Dynamics**
  The environment supports both:
  - Simulation mode, which models altitude dynamics with actuation lag and additive noise, and
  - Real-hardware mode, which interfaces directly with the Crazyflie flight stack and executes velocity commands through the onboard motion controller.
    
- **Safety and Termination Conditions**  
  Episodes terminate upon exceeding predefined altitude limits, violating roll or pitch safety bounds (real hardware), or reaching the maximum number of steps. In real-hardware mode, emergency stop and landing procedures are automatically triggered to ensure safe operation.
  

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
│   └── Environment definition (simulation and hardware interface)
│
├── train.py
│   └── Main training script
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
```

*(Contents may vary slightly depending on configuration.)*

---

## Installation

### System Requirements
- Linux operating system
- Python ≥ 3.9
- Bitcraze Crazyflie 2.1 (with Crazyradio PA)

It is recommended to use a Python virtual environment for installation.

```bash
pip install -r requirements.txt
```

This environment uses the **Bitcraze Crazyflie Python library (cflib)**. Ensure that the Crazyflie drivers and any required USB radio drivers are installed.

---

## Hardware Setup and Verification

Before running any RL experiments, verify the connection to the Crazyflie:

```bash
cfclient
```

Ensure that:
- the Crazyradio PA dongle is plugged in and detected
- the Crazyflie connects successfully
- onboard sensor readings update correctly (via the client software)

Also verify that the Flow Deck is attached and functioning. Refer to Bitcraze documentation for detailed setup and troubleshooting.

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

### Example – Setting target altitude and action limits
```bash
python train.py --task real --target_altitude 1.0 --action_range 0.20
```

Configuration parameters (e.g., target altitude or action range) can also be adjusted in the argument definitions under `utils/`.

---

## Safety Notes

- These experiments involve **real flying hardware**.
- Always operate in a clear, enclosed area.
- Use conservative action limits for initial testing.
- Never deploy untested policies directly on hardware.

---

## Reproducibility

- Each experiment was repeated across multiple flight trials to ensure consistency.
- Real-world variability (battery levels, sensor drift, etc.) can cause performance differences between runs.
- We focus on **relative improvements** in control smoothness and stability rather than exact numerical replication of results.

---

## Relation to the Paper

The quadcopter experiments serve as a **hardware validation** of SAPPS, demonstrating that the method achieves smoother control and improved stability in a physical system beyond what is observed in simulation benchmarks.

---

## Citation

If you use this environment or experimental setup, please cite:

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

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and by the **National Research Council Canada (NRC)**. The implementation builds upon the Bitcraze Crazyflie ecosystem and its open-source tools.
