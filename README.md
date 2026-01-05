# SAPPS-RL
## Adaptive Policy Regularization for Smooth Control in Reinforcement Learning

This repository contains the official implementation of **State-Adaptive Proportional Policy Smoothing (SAPPS)**, a policy regularization method designed to produce **smooth yet responsive control policies** in continuous-control reinforcement learning.

SAPPS suppresses high-frequency oscillations in learned policies **without compromising performance**, particularly in **dynamic environments** where rapid adaptation is required.

📄 **Paper**: *Adaptive Policy Regularization for Smooth Control in Reinforcement Learning*  
📌 **Status**: Under review  
👤 **Authors**: Payam Parvizi, Abhishek Naik, Colin Bellinger, Ross Cheriton, and Davide Spinello   
🔗 **Repository**: https://github.com/payamparvizi/sapps-rl  

---

## Abstract (Paper Summary)

A significant challenge in applying reinforcement learning (RL) to continuous-control problems is the presence of high-frequency oscillations in the actions produced by learned policies. These oscillations result in abrupt control responses, which can lead to excessive actuator wear, increased power consumption, and instability in real-world deployments. Existing approaches designed to reduce such oscillations often involve trade-offs, including increased architectural complexity or degraded policy performance. In particular, performance degradation tends to occur in environments where rapid state changes require rapid adaptation. To address this issue, we propose the State-Adaptive Proportional Policy Smoothing (SAPPS), a novel approach that adaptively adjusts smoothness constraints to suppress high-frequency components in RL policies. SAPPS utilizes Lipschitz continuity during policy optimization to learn a control response such that changes in consecutive actions scale proportionally with changes in consecutive observations, enabling smooth yet responsive control. Results from simulations and hardware implementation demonstrate that SAPPS can produce smooth control policies without compromising performance across a diverse set of environments, including MuJoCo continuous-control tasks, a simulated adaptive optics system for optical satellite communications, and a real-world nano quadcopter, under both slowly and rapidly changing conditions.

---

## Method Overview

SAPPS method that can be integrated into deep RL algorithms to improve policy smoothness in both static and dynamic continuous-control environments. SAPPS regularizes changes between consecutive actions based on the relative magnitude of variation between temporally consecutive observations, encouraging proportional action responses. This adaptive design penalizes unnecessary action fluctuations while preserving responsiveness to meaningful observation changes.

SAPPS is compared against:
- Vanilla PPO
- PPO with Conditioning for Action Policy Smoothness (CAPS)
- PPO with LipsNet-based architectures

---

## Repository Structure
sapps-rl/
├── Adaptive_Optics_Environment/
│ └── RL environment for wavefront sensorless adaptive optics
│
├── MuJoCo_Environments/
│ └── Standard MuJoCo continuous-control benchmark tasks
│
├── Quadcopter_Environment/
│ └── Real-world nano quadcopter hovering experiments
│
└── README.md
