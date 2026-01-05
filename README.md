# SAPPS-RL
## Adaptive Policy Regularization for Smooth Control in Reinforcement Learning

This repository contains the official implementation of **State-Adaptive Proportional Policy Smoothing (SAPPS)**, a policy regularization method designed to produce **smooth yet responsive control policies** in continuous-control reinforcement learning.

SAPPS suppresses high-frequency oscillations in learned policies **without compromising performance**, particularly in **dynamic environments** where rapid adaptation is required.

📄 **Paper**: *Adaptive Policy Regularization for Smooth Control in Reinforcement Learning*  
📌 **Status**: Under review  
👤 **Author**: Payam Parvizi  
🔗 **Repository**: https://github.com/payamparvizi/sapps-rl  

---

## Abstract (Paper Summary)

A significant challenge in applying reinforcement learning (RL) to continuous-control problems is the presence of high-frequency oscillations in the actions produced by learned policies. These oscillations lead to abrupt control responses, excessive actuator usage, increased power consumption, and instability in real-world systems. Existing smoothing and regularization approaches often involve trade-offs, including increased architectural complexity or degraded performance, particularly in environments that require rapid adaptation.

To address this challenge, **State-Adaptive Proportional Policy Smoothing (SAPPS)** is proposed. SAPPS introduces a state-adaptive regularization mechanism that suppresses high-frequency policy components by encouraging changes in consecutive actions to scale proportionally with changes in consecutive observations. This proportional behavior enables smooth control in quasi-static conditions while preserving responsiveness in highly dynamic environments.

SAPPS is integrated with Proximal Policy Optimization (PPO) and evaluated across a diverse set of environments, including MuJoCo continuous-control benchmarks, a simulated wavefront sensorless adaptive optics system for optical satellite communications, and a real-world nano quadcopter. Results demonstrate that SAPPS consistently improves policy smoothness without degrading task performance.

---

## Key Idea

**Smoothness should depend on how much the state changes.**

Instead of enforcing fixed action penalties or architectural constraints, SAPPS introduces a **state-adaptive proportional regularization** that:

- penalizes **large action changes** when **state changes are small**
- penalizes **small action changes** when **state changes are large**

This mechanism is **inspired by Lipschitz continuity**, but operates as a **local, state-transition-conditioned constraint**, rather than enforcing global Lipschitz bounds.

---

## Method Overview

SAPPS augments the PPO objective with an additional regularization term that relates differences between consecutive actions to differences between consecutive observations. This formulation suppresses high-frequency oscillations while allowing rapid control responses when the environment changes significantly.

SAPPS is compared against:
- Vanilla PPO
- PPO with Conditioning for Action Policy Smoothness (CAPS)
- PPO with LipsNet-based architectures

---

## Repository Structure

