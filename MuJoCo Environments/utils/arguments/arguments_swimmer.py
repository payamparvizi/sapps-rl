import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Swimmer-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=4096)
    # parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--hidden_size", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=8)
    parser.add_argument("--test-num", type=int, default=10)

    parser.add_argument("--rew-norm", type=int, default=True)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    
    parser.add_argument("--regularization_case", type=str, default="standard_PPO")
    
    # CAPS parameters
    parser.add_argument("--lambda_T", type=float, default=1e-3)
    parser.add_argument("--lambda_S", type=float, default=5e-3)
    parser.add_argument("--sigma_s_bar", type=float, default=100)
    
    # APS parameters
    parser.add_argument("--lambda_P", type=float, default=1e-4)
    parser.add_argument("--c_homog", type=float, default=50)
    parser.add_argument("--noise_pym", type=float, default=1e-12)
    
    parser.add_argument("--length_sm", type=int, default=1000)
    parser.add_argument("--episodes-sm", type=int, default=1)
    parser.add_argument("--episodes-sm-final", type=int, default=100)
    
    return parser.parse_args()
