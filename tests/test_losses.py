import unittest
import torch
from src.losses import FocalLoss

class TestFocalLoss(unittest.TestCase):
    def test_numerical_stability(self):
        """
        Verify that FocalLoss does not NaN or overflow under extreme outputs.
        """
        criterion = FocalLoss(gamma=2.0)
        
        # Test case 1: Very high probability (well-classified, loss -> 0)
        logits_easy = torch.tensor([[100.0, -100.0]], dtype=torch.float32)
        targets_easy = torch.tensor([0], dtype=torch.long)
        loss_easy = criterion(logits_easy, targets_easy)
        self.assertFalse(torch.isnan(loss_easy))
        self.assertLess(loss_easy.item(), 1e-4)

        # Test case 2: Very low probability (highly anomalous, loss is penalized)
        logits_hard = torch.tensor([[-100.0, 100.0]], dtype=torch.float32)
        targets_hard = torch.tensor([0], dtype=torch.long)
        loss_hard = criterion(logits_hard, targets_hard)
        self.assertFalse(torch.isnan(loss_hard))
        self.assertGreater(loss_hard.item(), 10.0)

    def test_ignore_index(self):
        """
        Verify that ignore_index is correctly filtered.
        """
        criterion = FocalLoss(gamma=2.0, ignore_index=-100)
        
        logits = torch.randn(2, 5, 256, requires_grad=True)
        targets = torch.tensor([[10, -100, 20, -100, 30],
                                [40, 50, -100, 60, -100]], dtype=torch.long)
        
        # Should not raise errors and run successfully
        loss = criterion(logits, targets)
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(loss.requires_grad)

    def test_gradient_flow(self):
        """
        Verify that gradients flow correctly through the custom loss to the input logits.
        """
        criterion = FocalLoss(gamma=2.0)
        
        logits = torch.randn(4, 10, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([2, 5, 0, 9], dtype=torch.long)
        
        loss = criterion(logits, targets)
        loss.backward()
        
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))

if __name__ == "__main__":
    unittest.main()
