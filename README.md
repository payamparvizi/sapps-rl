# SAPPS-RL
## Adaptive Policy Regularization for Smooth Control in Reinforcement Learning

This repository contains the official implementation of **State-Adaptive Proportional Policy Smoothing (SAPPS)**, a policy regularization method designed to produce **smooth yet responsive control policies** in continuous-control reinforcement learning.

SAPPS suppresses high-frequency oscillations in learned policies **without compromising performance**, particularly in **dynamic environments** where rapid adaptation is required.

📄 **Paper**: *Adaptive Policy Regularization for Smooth Control in Reinforcement Learning*  
📌 **Journal submission**: IEEE Transactions on Automation Science and Engineering (under review)  
🔗 **Preprint**: arXiv (forthcoming)  
👤 **Authors**: [Payam Parvizi](https://www.linkedin.com/in/payamparvizi/), Abhishek Naik, Colin Bellinger, Ross Cheriton, Davide Spinello

---

## Abstract

A significant challenge in applying reinforcement learning (RL) to continuous-control problems is the presence of high-frequency oscillations in the actions produced by learned policies. These oscillations result in abrupt control responses, leading to excessive actuator wear, increased power consumption, and instability in real-world deployments. Existing approaches to reduce such oscillations often involve trade-offs, including increased architectural complexity or degraded policy performance, particularly in environments where rapid state changes require rapid adaptation.

To address this issue, we propose **State-Adaptive Proportional Policy Smoothing (SAPPS)**, a novel approach that adaptively adjusts smoothness constraints to suppress high-frequency components in RL policies. SAPPS is inspired by Lipschitz continuity. It introduces a state-adaptive proportional regularization during policy optimization, encouraging changes in consecutive actions to scale with changes in consecutive observations. This adaptive constraint enables smooth yet responsive control.

Results from simulation and hardware experiments demonstrate that SAPPS produces smooth control policies without compromising performance across a diverse set of environments, including MuJoCo continuous-control benchmarks, a simulated adaptive optics system for optical satellite communications, and a real-world nano quadcopter, under both slowly and rapidly changing conditions.

---


## Method Overview

SAPPS is a general policy regularization technique that can be integrated into deep RL algorithms to improve policy smoothness in both static and dynamic continuous-control settings. Rather than directly penalizing action magnitude, SAPPS regularizes the change between consecutive actions based on the relative change between consecutive observations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3b4df3bc-2d37-49bb-ba33-a41ac0952c8d" align="center" width="400">
</p>

This adaptive formulation:
- penalizes unnecessary action fluctuations when state changes are small
- preserves responsiveness when large observation changes require rapid control adaptation

SAPPS is implemented within **Proximal Policy Optimization (PPO)** and compared against:
- Vanilla PPO  
- PPO with Conditioning for Action Policy Smoothness (CAPS)  
- PPO with LipsNet-based architectures  

---

## Repository Structure

```
sapps-rl/
│
├── Adaptive_Optics_Environment/
├── MuJoCo_Environments/
├── Quadcopter_Environment/
└── README.md
```

Each environment directory is self-contained and includes training and evaluation scripts corresponding to the experiments reported in the paper.

---

## Experimental Domains

### 1. MuJoCo Continuous-Control Benchmarks
SAPPS is evaluated on standard OpenAI Gymnasium MuJoCo tasks, including:
- Walker2D  
- HalfCheetah  
- Ant  
- Reacher  
- Swimmer  

The diagnostic experiment in the paper is conducted on a continuing version of the Reacher task, whose implementation is straightforward and closely follows that in [this repository](https://github.com/abhisheknaik96/csuite/blob/continuing_mujoco/csuite/environments/mujoco_continuing/envs_all.py#L116).

Across these benchmarks, SAPPS improves policy smoothness while maintaining or improving task return.

### 2. Real-World Quadcopter Control
A nano quadcopter hovering task is used to validate real-world applicability. SAPPS demonstrates reduced control oscillations, improved actuator efficiency, and stable performance under sensor noise and disturbances.

### 3. Wavefront Sensorless Adaptive Optics
An optical control problem for satellite-to-ground optical communication. SAPPS is evaluated under both slowly varying and rapidly varying atmospheric turbulence, achieving consistently strong performance and improved robustness relative to benchmark policy smoothing methods.

---

## Installation

All experiments are implemented in Python and use standard deep reinforcement learning libraries.

### Requirements
- Python ≥ 3.9  
- PyTorch  
- NumPy
- SciPy
- Farama Gymnasium (with MuJoCo support)
- Weights & Biases (for logging)
- tianshou (training framework and rollout collection)

Each environment subdirectory includes its own `requirements.txt` listing any additional dependencies (e.g., specialized simulation libraries or hardware interface packages).

---

## Quick Start (Minimal Example)

Below is a minimal example to train a SAPPS-regularized PPO policy on a MuJoCo task.

```bash
# 1. Create and activate a virtual environment
python -m venv sapps-env
source sapps-env/bin/activate  # Windows: sapps-env\Scripts\activate

# 2. Install dependencies for MuJoCo experiments
cd MuJoCo_Environments
pip install -r requirements.txt

# 3. Run a minimal training example (default environment: Ant-v4)
python run_mujoco.py \
  --regularization_case PPO_SAPPS \
  --seed 0
```

By default, this command trains on the Ant-v4 environment using the hyperparameters reported in the paper.

---

## Running Experiments

Each environment directory contains its own training and evaluation scripts. Please refer to the specific environment's README and scripts for details on usage, hyperparameters, and experimental settings.

---

## Reproducibility

All results reported in the paper are averaged over multiple random seeds, and hyperparameters match those described in the paper. While evaluation protocols are consistent across methods, differences in simulator versions, hardware, or inherent randomness may cause your results to vary slightly. However, the qualitative performance trends should remain consistent.

---

## Citation

If you use this code in your research, please cite the associated paper.

🔗 **Preprint**: arXiv (forthcoming)

A full BibTeX entry and `CITATION.cff` file will be added upon publication.

---

## License

This project is released under the MIT License. See the LICENSE file for details.

---

## Contact

**Payam Parvizi**  
Email: pparv056@uottawa.ca  
GitHub: https://github.com/payamparvizi

---

## Acknowledgments

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and by the **National Research Council Canada (NRC)**.
