import pytest
import torch

from src.pde.multidiff import Multidiff, MultidiffNetwork
from src.util.equality import EqualityTorch


class TestMultidiff:
    def _lhs_default(self) -> torch.Tensor:
        return torch.tensor([1, 3, 7], dtype=torch.float32)

    def test_ctor_rhs_preprocessing(self):
        for rhs in [
            torch.tensor(42),  # shape = []
            torch.tensor([42]),  # shape = [1]
            torch.tensor([[42]]),  # shape = [1, 1]
            torch.tensor([[[42]]]),  # shape = [1, 1, 1]
            torch.tensor([10, 20]),  # shape = [2]
            torch.tensor([10, 20, 30]),  # shape = [3]
            torch.tensor([[1, 2, 3], [10, 20, 30]]),  # shape = [2, 3]
        ]:
            assert Multidiff(rhs, self._lhs_default())

        with pytest.raises(ValueError):
            for rhs in [
                torch.tensor([10, 20]),  # shape = [2]
                torch.tensor([10, 20, 30]),  # shape = [3]
                torch.tensor([[1, 2, 3], [10, 20, 30]]),  # shape = [2, 3]
            ]:
                Multidiff(rhs, self._lhs_default(), force_rhs_scalar=True)

    def test_diff_const(self):
        rhs = torch.tensor(42)
        assert torch.equal(
            Multidiff(rhs, self._lhs_default()).diff(),
            torch.zeros_like(self._lhs_default()),
        )

        lhs = self._lhs_default()  # |requires_grad| is FALSE by default
        md = Multidiff(torch.dot(lhs, lhs), lhs)
        for _ in range(3):
            assert torch.equal(md.diff(), torch.tensor([0, 0, 0]))

    def test_diff_nonconst(self):
        lhs = self._lhs_default().requires_grad_(True)
        # rhs = 2 x1^2 + 3 x2^2 + 4 x3^2
        rhs = torch.squeeze(torch.dot(lhs**2, torch.linspace(2, 4, steps=3)))

        md = Multidiff(rhs, lhs)
        assert torch.equal(
            md.diff(),
            torch.tensor([4, 18, 56]),  # 4 x1 | 6 x2 | 8 x3
        )
        assert torch.equal(md.diff(), torch.tensor([4, 6, 8]))
        assert torch.equal(md.diff(), torch.tensor([0, 0, 0]))
        assert torch.equal(md.diff(), torch.tensor([0, 0, 0]))

    def test_diff_mix_const_nonconst(self):
        lhs = self._lhs_default().requires_grad_(True)
        # rhs = 2 x1^2 + 3 x2^2
        rhs = torch.squeeze(torch.dot(lhs**2, torch.tensor([2.0, 3.0, 0.0])))

        md = Multidiff(rhs, lhs)
        assert torch.equal(
            md.diff(),
            torch.tensor([4, 18, 0]),  # 4 x1 | 6 x2 | 0
        )
        assert torch.equal(md.diff(), torch.tensor([4, 6, 0]))
        assert torch.equal(md.diff(), torch.tensor([0, 0, 0]))
        assert torch.equal(md.diff(), torch.tensor([0, 0, 0]))


class TestMultidiffNetwork:
    _size_input, _size_hidden, _size_output = 3, 7, 2

    def _make_network(self, activation: torch.nn.Module):
        return torch.nn.Sequential(
            torch.nn.Linear(self._size_input, self._size_hidden),
            activation(),
            torch.nn.Linear(self._size_hidden, self._size_output),
        )

    def test_network_with_sigmoid(self):
        torch.manual_seed(42)
        mdn = MultidiffNetwork(
            self._make_network(torch.nn.Sigmoid),
            torch.rand([self._size_input], requires_grad=True),
        )

        assert EqualityTorch(
            mdn.diff(0),
            torch.tensor([0.2385, 0.6690]),
        ).is_close()
        assert EqualityTorch(
            mdn.diff(1),
            torch.tensor([0.0024, 0.0724, 0.0775]),
        ).is_close()
        assert EqualityTorch(
            mdn.diff(2),
            torch.tensor([0.0038, -0.0142, -0.0152]),
        ).is_close()
        assert EqualityTorch(
            mdn.diff(3),
            torch.tensor([-0.0042, -0.0178, -0.0088]),
        ).is_close()

    def test_network_with_relu(self):
        torch.manual_seed(42)
        mdn = MultidiffNetwork(
            self._make_network(torch.nn.ReLU),
            torch.rand([self._size_input], requires_grad=True),
        )

        assert EqualityTorch(
            mdn.diff(0),
            torch.tensor([0.3960, 0.5875]),
        ).is_close()
        [-0.1561, 0.4016, 0.3042]
        assert EqualityTorch(
            mdn.diff(1),
            torch.tensor([-0.1561, 0.4016, 0.3042]),
        ).is_close()
        assert EqualityTorch(mdn.diff(2), torch.zeros([3])).is_equal()
        assert EqualityTorch(mdn.diff(3), torch.zeros([3])).is_equal()
