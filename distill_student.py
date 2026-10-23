# distill_student.py
#student model & standalone distillation utilities

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32
DEVICE = "cpu"


class StudentPolicyDiscrete(nn.Module):
    """
    Small MLP
    outputs a categorical distribution over actions,,
    Used as the distilled student policy 0w^
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


def distill_train(
    model: StudentPolicyDiscrete,
    X: torch.Tensor,
    P: torch.Tensor,
    steps: int = 300,
    lr: float = 5e-3,
    beta_reg: float = 1e-4,
):
    """
    Minimize KL(teacher || student) + L2 regularization.
    X: (N, obs_dim) student inputs
    P: (N, n_actions) teacher probabilities
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for it in range(steps):
        probs, _ = model(X)
        # KL(teacher || student)
        kl = torch.sum(
            P * (torch.log(P + 1e-8) - torch.log(probs + 1e-8)), dim=1
        ).mean()
        reg = sum((p ** 2).sum() for p in model.parameters())
        loss = kl + beta_reg * reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (it + 1) % 50 == 0:
            print(
                f"[distill] step {it+1:3d} "
                f"loss={loss.item():.4f} kl={kl.item():.4f}"
            )

    return model


def main():
    parser = argparse.ArgumentParser(description="Distill student policy from teacher data")
    parser.add_argument("--data-X", type=str, default="teacher_data_X.npy", help="Teacher input features")
    parser.add_argument("--data-P", type=str, default="teacher_data_P.npy", help="Teacher action probabilities")
    parser.add_argument("--steps", type=int, default=300, help="Distillation training steps")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--beta-reg", type=float, default=1e-4, help="L2 regularization coefficient")
    parser.add_argument("--output", type=str, default="student_policy_init.pt", help="Output checkpoint path")
    args = parser.parse_args()
    
    print(">>> distill_student.py: starting main()")  # debug print

    # Load teacher dataset
    X_np = np.load(args.data_X)
    P_np = np.load(args.data_P)

    X = torch.tensor(X_np, dtype=DTYPE, device=DEVICE)
    P = torch.tensor(P_np, dtype=DTYPE, device=DEVICE)

    obs_dim = X.shape[1]
    n_actions = P.shape[1]

    print(f"Loaded dataset: X shape {X.shape}, P shape {P.shape}")
    print(f"Inferred: obs_dim={obs_dim}, n_actions={n_actions}")

    # Build student and distill
    student = StudentPolicyDiscrete(obs_dim, n_actions).to(DEVICE)
    student = distill_train(student, X, P, steps=args.steps, lr=args.lr, beta_reg=args.beta_reg)

    # Save weights for PPO warm-start
    torch.save(student.state_dict(), args.output)
    print(f"Saved distilled student to {args.output}")


if __name__ == "__main__":   # IMPORTANT: this must be exactly like this
    main()
