from types import SimpleNamespace

import pytest
import torch

from nanodiloco.training_utils import forward_backward


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x, y):
        return SimpleNamespace(loss=torch.nn.functional.mse_loss(self.linear(x), y))


@pytest.mark.parametrize("accumulation_steps", [1, 2, 4, 8])
def test_accumulated_gradients_match_full_batch(accumulation_steps):
    torch.manual_seed(0)
    x, y = torch.randn(8, 4), torch.randn(8, 1)

    full = TinyModel()
    accumulated = TinyModel()
    accumulated.load_state_dict(full.state_dict())

    full_loss = forward_backward(full, {"x": x, "y": y}, gradient_accumulation_steps=1)

    micro_losses = [
        forward_backward(accumulated, {"x": mx, "y": my}, accumulation_steps)
        for mx, my in zip(x.chunk(accumulation_steps), y.chunk(accumulation_steps))
    ]

    for p_full, p_acc in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(p_acc.grad, p_full.grad)
    # Returned loss is the unscaled micro-batch loss, so their mean is the full-batch loss.
    torch.testing.assert_close(torch.stack(micro_losses).mean(), full_loss)
