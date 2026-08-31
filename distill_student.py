"""
Distillation process for the student policy in the discrete fixture environment.
This module contains functions and classes necessary for training the student model
using knowledge distillation techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim

class StudentPolicyDiscrete(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super(StudentPolicyDiscrete, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        probs = nn.functional.softmax(logits, dim=-1)
        return probs, logits

def distill_train(student_model, teacher_probs, obs, steps=300, lr=5e-3, kl_weight=1.0):
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    for step in range(steps):
        optimizer.zero_grad()
        student_probs, logits = student_model(obs)
        
        # KL Divergence loss
        kl_loss = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-12) - torch.log(student_probs + 1e-12)), dim=1).mean()
        
        # Cross-entropy loss
        hard_labels = torch.argmax(teacher_probs, dim=1)
        cross_entropy = ce_loss(logits, hard_labels)

        # Total loss
        loss = kl_weight * kl_loss + cross_entropy
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"[Distillation] Step {step + 1}/{steps} | Loss: {loss.item():.4f} | KL: {kl_loss.item():.4f}")

    return student_model