from typing import Callable

import pytest
import torch

from src.util.equality import EqualityTorch
from src.util.multidiff import Multidiff, MultidiffNetwork


class _Data:
    def __init__(self):
        self._lhs_raw = [
            [1, 2, 3],
            [1, 2, 3],
            [3, 4, 5],
            [10, 20, 30],
            [10, 20, 30],
            [0, 0, 0],
        ]

    def lhs(self, grad: bool = True) -> torch.Tensor:
        return torch.tensor(self._lhs_raw, dtype=torch.float32, requires_grad=grad)

    def operator_product_of_squares(self) -> Callable:
        class Network:
            def __call__(self, lhs: torch.Tensor) -> torch.Tensor:
                return torch.prod(lhs**2, dim=1).view(-1, 1)

        return Network()


class TestMultidiff:
    def test_ctor_preprocessing(self):
        # 1 data-pair
        for rhs in [
            torch.tensor(42),  # shape = []
            torch.tensor([42]),  # shape = [1]
            torch.tensor([[42]]),  # shape = [1, 1]
        ]:
            assert Multidiff(rhs, torch.rand((1, 3)))

        for size_rhs, size_lhs in [
            ((1, 1), (1, 1)),
            ((1, 1), (1, 3)),
            ((20, 1), (20, 3)),
            ((20), (20, 3)),  # (rhs) missing dim auto-filled
            ((20, 1), (20)),  # (lhs) missing dim auto-filled
            ((20), (20)),  # (rhs&lhs) missing dim auto-filled
        ]:
            assert Multidiff(torch.rand(size_rhs), torch.rand(size_lhs))

        with pytest.raises(ValueError):
            for size_rhs, size_lhs in [
                ((20, 1, 1), (20, 3)),  # rhs too many dims
                ((20, 1), (20, 3, 3)),  # lhs too many dims
                ((20, 2), (20, 3)),  # rhs.second-dim > 1
                ((19, 1), (20, 3)),  # len(rhs) != len(lhs)
            ]:
                Multidiff(torch.rand(size_rhs), torch.rand(size_lhs))

    def test_diff_const(self):
        lhs = _Data().lhs()

        rhs = torch.tensor(42).repeat(len(lhs), 1)
        assert torch.equal(Multidiff(rhs, lhs).diff(), torch.zeros_like(lhs))

        lhs.requires_grad = False
        rhs = torch.sum(2 * lhs, dim=1)
        md = Multidiff(rhs, lhs)
        for _ in range(3):
            assert torch.equal(md.diff(), torch.zeros_like(lhs))

    def test_diff_nonconst(self):
        lhs = _Data().lhs(grad=True)
        # rhs[some_row, :] = x1^2 * x2^2 * x3^2
        rhs = _Data().operator_product_of_squares()(lhs)
        md = Multidiff(rhs, lhs)

        # diff[some_row, :] = 2 (x2^2 x3^2) x1 | 2 (x1^2 x3^2) x2 | 2 (x2^2 x3^2) x1
        assert torch.equal(
            md.diff(),
            torch.tensor(
                [
                    [72, 36, 24],
                    [72, 36, 24],
                    [2400, 1800, 1440],
                    [7200000, 3600000, 2400000],
                    [7200000, 3600000, 2400000],
                    [0, 0, 0],
                ]
            ),
        )


class TestMultidiffNetwork:
    _size_input, _size_hidden, _size_output = 3, 7, 1

    def _make_network(self, activation: torch.nn.Module):
        return torch.nn.Sequential(
            torch.nn.Linear(self._size_input, self._size_hidden),
            activation(),
            torch.nn.Linear(self._size_hidden, self._size_output),
        )

    def test_network_raw(self):
        data = _Data()
        lhs = data.lhs()
        network = data.operator_product_of_squares()
        mdn = MultidiffNetwork(network, lhs, ["t", "x1", "x2"])

        assert EqualityTorch(
            mdn.diff_0(),
            torch.tensor(
                [
                    [36],
                    [36],
                    [3600],
                    [36000000],
                    [36000000],
                    [0],
                ]
            ),
        ).is_equal()

        assert EqualityTorch(
            mdn.diff("t", 1),
            torch.tensor(
                [
                    [72],
                    [72],
                    [2400],
                    [7200000],
                    [7200000],
                    [0],
                ]
            ),
        ).is_equal()
        assert EqualityTorch(
            mdn.diff("t", 2),
            torch.tensor(
                [
                    [72],
                    [72],
                    [800],
                    [720000],
                    [720000],
                    [0],
                ]
            ),
        ).is_equal()
        assert EqualityTorch(mdn.diff("t", 3), torch.zeros((len(lhs), 1))).is_equal()
        assert EqualityTorch(mdn.diff("t", 4), torch.zeros((len(lhs), 1))).is_equal()

        assert EqualityTorch(
            mdn.diff("x1", 1),
            torch.tensor(
                [
                    [36],
                    [36],
                    [1800],
                    [3600000],
                    [3600000],
                    [0],
                ]
            ),
        ).is_equal()
        assert EqualityTorch(
            mdn.diff("x1", 2),
            torch.tensor(
                [
                    [18],
                    [18],
                    [450],
                    [180000],
                    [180000],
                    [0],
                ]
            ),
        ).is_equal()
        assert EqualityTorch(mdn.diff("x1", 3), torch.zeros((len(lhs), 1))).is_equal()
        assert EqualityTorch(mdn.diff("x1", 4), torch.zeros((len(lhs), 1))).is_equal()

        assert EqualityTorch(
            mdn.diff("x2", 1),
            torch.tensor(
                [
                    [24],
                    [24],
                    [1440],
                    [2400000],
                    [2400000],
                    [0],
                ]
            ),
        ).is_equal()
        assert EqualityTorch(
            mdn.diff("x2", 2),
            torch.tensor(
                [
                    [8],
                    [8],
                    [288],
                    [80000],
                    [80000],
                    [0],
                ]
            ),
        ).is_equal()
        assert EqualityTorch(mdn.diff("x2", 3), torch.zeros((len(lhs), 1))).is_equal()
        assert EqualityTorch(mdn.diff("x2", 4), torch.zeros((len(lhs), 1))).is_equal()

    def test_network_with_sigmoid(self):
        torch.manual_seed(42)
        lhs = _Data().lhs()
        mdn = MultidiffNetwork(
            self._make_network(torch.nn.Sigmoid),
            lhs,
            lhs_names=["t", "x1", "x2"],
        )

        assert EqualityTorch(
            mdn.diff_0(),
            torch.tensor(
                [
                    [0.3871],
                    [0.3871],
                    [0.3846],
                    [0.2718],
                    [0.2718],
                    [0.2690],
                ]
            ),
        ).is_close()

        assert EqualityTorch(
            mdn.diff("t", 1),
            torch.tensor(
                [
                    [0.0078],
                    [0.0078],
                    [0.0066],
                    [-0.0004],
                    [-0.0004],
                    [0.0016],
                ]
            ),
        ).is_close()

        assert EqualityTorch(
            mdn.diff("t", 2),
            torch.tensor(
                [
                    [-0.0004],
                    [-0.0004],
                    [0.0101],
                    [0.0003],
                    [0.0003],
                    [-0.0059],
                ]
            ),
        ).is_close()

        assert EqualityTorch(
            mdn.diff("x1", 1),
            torch.tensor(
                [
                    [-7.3901e-03],
                    [-7.3901e-03],
                    [-1.9473e-02],
                    [7.8624e-05],
                    [7.8624e-05],
                    [1.3728e-02],
                ]
            ),
        ).is_close()

        assert EqualityTorch(
            mdn.diff("x1", 2),
            torch.tensor(
                [
                    [-3.4731e-04],
                    [-3.4731e-04],
                    [4.2486e-03],
                    [2.7648e-05],
                    [2.7648e-05],
                    [-5.9944e-03],
                ]
            ),
        ).is_close()

        assert EqualityTorch(
            mdn.diff("x2", 1),
            torch.tensor(
                [
                    [0.0166],
                    [0.0166],
                    [-0.0002],
                    [-0.0001],
                    [-0.0001],
                    [0.0509],
                ]
            ),
        ).is_close()

        assert EqualityTorch(
            mdn.diff("x2", 2),
            torch.tensor(
                [
                    [-8.5902e-03],
                    [-8.5902e-03],
                    [-3.6584e-03],
                    [9.9641e-06],
                    [9.9641e-06],
                    [-3.9901e-03],
                ]
            ),
        ).is_close()

    def test_network_with_relu(self):
        torch.manual_seed(42)
        lhs = _Data().lhs()
        mdn = MultidiffNetwork(
            self._make_network(torch.nn.ReLU),
            lhs,
            lhs_names=["t", "x1", "x2"],
        )

        assert EqualityTorch(
            mdn.diff_0(),
            torch.tensor(
                [
                    [1.1564],
                    [1.1564],
                    [1.6048],
                    [6.9938],
                    [6.9938],
                    [0.3643],
                ]
            ),
        ).is_close()

        assert EqualityTorch(
            mdn.diff("t", 1),
            torch.tensor(
                [
                    [-0.0229],
                    [-0.0229],
                    [-0.0229],
                    [-0.0229],
                    [-0.0229],
                    [-0.0959],
                ]
            ),
        ).is_close()
        assert EqualityTorch(
            mdn.diff("x1", 1),
            torch.tensor(
                [
                    [0.0697],
                    [0.0697],
                    [0.0697],
                    [0.0697],
                    [0.0697],
                    [0.1157],
                ]
            ),
        ).is_close()
        assert EqualityTorch(
            mdn.diff("x2", 1),
            torch.tensor(
                [
                    [0.1773],
                    [0.1773],
                    [0.1773],
                    [0.1773],
                    [0.1773],
                    [0.1737],
                ]
            ),
        ).is_close()

        for target in ["t", "x1", "x2"]:
            for order in [2, 3, 4]:
                assert EqualityTorch(
                    mdn.diff(target, order),
                    torch.tensor([0] * len(lhs), dtype=torch.float).view(-1, 1),
                ).is_equal()
