"""
fin/meta/train_meta.py
Gradient-based meta-learning loop for the topology controller.

True meta-learning: the topology policy MLP is updated by Adam to
minimise the dissonance objective — no random selection logic.
"""

import torch
from fin.core.fin_system import FINSystem


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_meta_controller(
    n_steps   : int   = 50,
    lr        : float = 3e-4,
    n_elements: int   = 64,
    device    : str   = "cpu",
):
    set_seed(42)
    model = FINSystem(n_elements=n_elements, device=device)
    model.to(device)
    optimiser = torch.optim.Adam(model.topology.parameters(), lr=lr)
    h       = torch.zeros(model.memory.hidden_dim, device=device)
    history = []

    for step in range(n_steps):
        optimiser.zero_grad()
        result = model.step(h, dt=0.05)
        loss   = result["dissonance"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.topology.parameters(), 1.0)
        optimiser.step()
        h = result["h"].detach()
        history.append(float(loss.detach()))

    return model, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_meta_controller(n_steps=50, device=device)
    print("Dissonance history (first/last 5):", history[:5], "...", history[-5:])
