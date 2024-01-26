import torch

from src.deepl.network import Network
from src.util.equality import EqualityTorch


class TestNetwork:
    def test_network_with_time(self):
        torch.manual_seed(42)
        network = Network(dim_x=1)

        assert EqualityTorch(
            network(torch.tensor([1, 2], dtype=torch.float)),
            torch.tensor([0.1550]),
        ).is_close()
        assert EqualityTorch(
            network(torch.tensor([[1, 2], [10, 20], [37, 42]], dtype=torch.float)),
            torch.tensor([[0.1550], [0.1694], [0.1298]]),
        ).is_close()

    def test_network_no_time(self):
        torch.manual_seed(42)
        network = Network(dim_x=1, with_time=False)

        assert EqualityTorch(
            network(torch.tensor([10], dtype=torch.float)),
            torch.tensor([-0.0477]),
        ).is_close()
        assert EqualityTorch(
            network(torch.tensor([[10], [20], [37]], dtype=torch.float)),
            torch.tensor([[-0.0477], [-0.0512], [-0.0510]]),
        ).is_close()

    def test_network_2d(self):
        torch.manual_seed(42)
        network = Network(dim_x=2)

        assert EqualityTorch(
            network(torch.rand(3)),
            torch.tensor([0.1549]),
        ).is_close()
        assert EqualityTorch(
            network(torch.rand((4, 3))),
            torch.tensor([[0.1493], [0.1520], [0.1580], [0.1517]]),
        ).is_close()
