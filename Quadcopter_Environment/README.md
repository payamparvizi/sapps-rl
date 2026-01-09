# Quadcopter Control Environment (Crazyflie 2.1)

This directory contains the **real-world quadcopter control environment** used to evaluate **State-Adaptive Proportional Policy Smoothing (SAPPS)** in the paper:

📄 **Paper**: *Adaptive Policy Regularization for Smooth Control in Reinforcement Learning*    
🔗 **Preprint**: arXiv (forthcoming)

Experiments are conducted on a **Bitcraze Crazyflie 2.1** nano quadcopter and demonstrate the effectiveness of SAPPS in producing **smooth, stable, and hardware-safe** control policies under real-world noise, delays, and disturbances.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e15c1612-ea9d-4a60-ac8f-029d1f8b9d1a" align="center" width="300">
</p>

---

## Overview

While simulation benchmarks are useful for controlled evaluation, real-world robotic systems introduce additional challenges such as sensor noise, actuator delays, communication latency, and unmodeled dynamics. These effects can significantly amplify high-frequency oscillations in learned policies.

This environment supports both **simulation-based training** and **real-world deployment** on a Crazyflie quadcopter. Quadcopter hovering is formulated as a **continuous-control Markov Decision Process (MDP)**, enabling training and deployment of reinforcement learning policies on physical hardware. It is designed to evaluate SAPPS’s impact on:

- control smoothness,
- actuator efficiency, and
- stability during real-world flight.


## Experimental Videos

The following videos present real-world quadcopter hovering experiments conducted on a **Crazyflie 2.1** platform.  
They qualitatively compare different policy regularization strategies under the same hardware conditions.

<table>
  <tr>
    <th align="center">PPO</th>
    <th align="center">CAPS</th>
    <th align="center">SAPPS (Proposed)</th>
  </tr>
  <tr>
    <td align="center">

https://github.com/user-attachments/assets/dcf9ec1b-819f-44d9-9880-120621b22abd

</td>
    <td align="center">

https://github.com/user-attachments/assets/8d5d55f9-7d96-42c6-8000-89fa2c1ab4c6

</td>
    <td align="center">

https://github.com/user-attachments/assets/0ae71900-ce90-476f-986e-f95c5cf01360

</td>
  </tr>
</table>

---

## Key Results

The plots below summarize altitude tracking and velocity behavior for **PPO**, **CAPS**, and **SAPPS (proposed method)** during the final episode of the real-world quadcopter experiment, where the objective is to reach and maintain a target altitude of 1 meter. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/79a0e66c-42e5-4391-8563-dd99bdb34968" width="1100">
</p>

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
├── crazyflie_env/
│   └── Crazyflie environment definition (simulation and real hardware)
│
├── ppo/
│   └── PPO agent implementation and policy/value network definitions
│
├── utils/
│   └── Argument definitions and control smoothness evaluation utilities
│
├── run_quadcopter.py
│   └── Entry point for training and evaluation
│
├── requirements.txt
│   └── Python dependencies for the quadcopter environment
│
└── README.md
```

---

## Installation

### System Requirements
- Linux operating system
- Python ≥ 3.9
- Bitcraze Crazyflie 2.1 (with Crazyradio PA)

It is recommended to use a Python virtual environment for installation.

### Python Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### Crazyflie Software Dependencies

This environment relies on the **Bitcraze Crazyflie Python library (`cflib`)** and the required USB radio drivers.

Detailed installation instructions for the Crazyflie software dependencies, including driver setup and hardware verification, are provided in the following repository: [https://github.com/payamparvizi/Crazyflie_RL](https://github.com/payamparvizi/Crazyflie_RL)

Please follow the setup steps in that repository before running experiments on real hardware.

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


***Notes:***
- Each experiment was repeated across multiple flight trials to ensure consistency.
- Real-world variability can cause performance differences between runs.

---

## Reproducibility

Due to real-world hardware effects (sensor noise, timing jitter, communication delays, and unmodeled disturbances), exact trajectory-level reproducibility is not guaranteed even with fixed random seeds. Results reported in the paper are aggregated over multiple runs.


---

## Relation to the Paper

The quadcopter experiments serve as a **hardware validation** of SAPPS, demonstrating that the method achieves smoother control and improved stability in a physical system beyond what is observed in simulation benchmarks.

---

## Citation

If you use this code in your research, please cite the associated paper.

🔗 **Preprint**: arXiv (forthcoming)

A full BibTeX entry and `CITATION.cff` file will be added upon publication.

---

## Acknowledgments

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and by the **National Research Council Canada (NRC)**. The implementation builds upon the Bitcraze Crazyflie platform and its open-source software.
