
# PyTorch: EvoSGD (SGD + periodic evolutionary step)

```python
import math
import torch
from torch.optim import Optimizer

class EvoSGD(Optimizer):
    """
    SGD with momentum/Nesterov/weight_decay + periodic evolutionary step.

    Evolution knobs:
      - evo_interval: every N optimizer steps, run an evolution cycle
      - pop_size: number of candidates (including current params as one candidate)
      - elite_frac: fraction of best candidates to keep for recombination
      - sigma: mutation stddev (relative to param scale); decays by sigma_decay after each cycle
      - recombine: "mean" to average elites

    Notes:
      - The evolutionary cycle *requires* a closure that returns the scalar loss with NO backward.
      - Normal SGD updates use gradients you compute outside (typical PyTorch pattern).
    """
    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        # evolution extras
        evo_interval=200,
        pop_size=8,
        elite_frac=0.25,
        sigma=0.02,
        sigma_decay=0.98,
        recombine="mean",
        reset_momentum_on_evo=True,
    ):
        if lr <= 0: raise ValueError("Invalid lr")
        if momentum < 0: raise ValueError("Invalid momentum")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov requires momentum > 0")
        if not (2 <= pop_size): raise ValueError("pop_size must be >= 2")
        if not (0 < elite_frac <= 0.5): raise ValueError("elite_frac in (0, 0.5]")
        if sigma <= 0: raise ValueError("sigma must be > 0")
        if recombine not in ("mean",):
            raise ValueError("Unsupported recombine")

        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        # global state
        self._t = 0
        self.evo_interval = evo_interval
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.recombine = recombine
        self.reset_momentum_on_evo = reset_momentum_on_evo

    @torch.no_grad()
    def step(self, closure=None):
        """Perform an SGD step; occasionally perform evolutionary search.

        closure: callable that returns current loss (forward only; no backward).
                 Required on evolutionary steps; optional otherwise.
        """
        loss = None

        # === 1) Standard SGD update (uses pre-computed gradients) ===
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)  # weight decay

                state = self.state[p]
                if mom != 0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(p)
                    buf.mul_(mom).add_(d_p)
                    state["momentum_buffer"] = buf
                    d = d_p.add(buf, alpha=mom) if nesterov else buf
                else:
                    d = d_p

                p.add_(d, alpha=-lr)

        self._t += 1

        # === 2) Periodic evolutionary step ===
        if self.evo_interval > 0 and (self._t % self.evo_interval == 0):
            if closure is None:
                raise RuntimeError("Evo step requires a forward-only closure returning loss.")

            # -- snapshot current parameters
            flat_params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
            base = [p.detach().clone() for p in flat_params]

            # utility to load a set of tensors into params
            def load_params(src_list):
                for p, s in zip(flat_params, src_list):
                    p.copy_(s)

            # scale of mutations: relative to parameter magnitude
            def param_scale(p):
                # Robust scale: sqrt(mean(p^2)) with floor
                s = torch.sqrt(torch.mean(p.pow(2))) if p.numel() > 0 else torch.tensor(0., device=p.device)
                return torch.clamp(s, min=1e-8)

            # evaluate base
            loss_vals = []
            candidates = []

            load_params(base)
            base_loss = float(closure().detach())
            loss_vals.append(base_loss)
            candidates.append(base)

            # generate and evaluate mutated candidates
            for _ in range(self.pop_size - 1):
                mutated = []
                for p in base:
                    s = param_scale(p) * self.sigma
                    noise = torch.randn_like(p) * s
                    mutated.append(p + noise)
                load_params(mutated)
                cand_loss = float(closure().detach())
                loss_vals.append(cand_loss)
                # store copy to avoid aliasing
                candidates.append([c.detach().clone() for c in mutated])

            # pick elites
            k = max(1, int(math.ceil(self.elite_frac * self.pop_size)))
            idx = torch.tensor(loss_vals).argsort()[:k].tolist()
            elites = [candidates[i] for i in idx]

            # recombine (mean of elites)
            if self.recombine == "mean":
                mean_params = []
                for parts in zip(*elites):
                    stacked = torch.stack(parts, dim=0)
                    mean_params.append(stacked.mean(dim=0))
                load_params(mean_params)

            # optional: reset momentum buffers after jumps
            if self.reset_momentum_on_evo:
                for p in flat_params:
                    st = self.state[p]
                    if "momentum_buffer" in st:
                        st["momentum_buffer"].zero_()

            # decay mutation scale
            self.sigma *= self.sigma_decay

            # return the *current* loss after recombination for logging
            loss = float(closure().detach())

        return loss
```

## How to use it

Here’s a minimal loop that does normal SGD every step and an evolution cycle every `evo_interval` steps. The closure runs *forward-only* (no backward) for quick fitness checks during the evolutionary phase.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# dummy regression
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
opt = EvoSGD(
    model.parameters(),
    lr=1e-2, momentum=0.9, nesterov=True, weight_decay=1e-4,
    evo_interval=100, pop_size=8, elite_frac=0.25, sigma=0.05, sigma_decay=0.97
)

X = torch.randn(512, 10)
true_w = torch.randn(10, 1)
y = X @ true_w + 0.1 * torch.randn(512, 1)

def loss_fn():
    pred = model(X)
    return F.mse_loss(pred, y)

for step in range(800):
    # standard gradient pass
    opt.zero_grad()
    loss = loss_fn()
    loss.backward()

    # closure for eval-only passes (no backward)
    def closure_eval_only():
        with torch.no_grad():
            return loss_fn()

    # EvoSGD uses grads for SGD, and occasionally calls the closure for evolution
    current = opt.step(closure=closure_eval_only)

    if step % 100 == 0:
        print(f"step {step:4d} | loss {float(loss):.4f}" + (f" | evo_loss {current:.4f}" if current is not None else ""))
```

## What “evolution” you just got (quick mapping)

* **Mutation**: add small Gaussian noise to parameters (scale ∝ parameter RMS × `sigma`).
* **Selection**: keep the candidates with lowest loss.
* **Recombination**: average the elite parameters.
* **Adaptation**: gradually shrink mutation (`sigma_decay`) as training progresses.
* **Exploit + Explore**: SGD exploits local gradient info every step; evolution explores new regions periodically.


