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

    def test_focal_loss_gamma_zero_equals_ce(self):
        """
        Verify that when gamma=0, Focal Loss is mathematically identical to standard Cross-Entropy Loss.
        """
        criterion_focal = FocalLoss(gamma=0.0, ignore_index=-100)
        criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        logits = torch.randn(4, 128, 256)
        targets = torch.randint(0, 256, (4, 128), dtype=torch.long)
        
        # Inject ignore_index padding values
        targets[0, 10:20] = -100
        targets[2, 50:70] = -100
        
        loss_focal = criterion_focal(logits, targets)
        
        # CrossEntropyLoss requires flattening
        logits_flat = logits.reshape(-1, 256)
        targets_flat = targets.reshape(-1)
        loss_ce = criterion_ce(logits_flat, targets_flat)
        
        self.assertTrue(torch.allclose(loss_focal, loss_ce, atol=1e-5))

    def test_focal_loss_all_padding_safe_exit(self):
        """
        Verify safe exit and gradient graph integrity when the entire batch is padding tokens.
        """
        criterion = FocalLoss(gamma=2.0, ignore_index=-100)
        
        logits = torch.randn(2, 10, 256, requires_grad=True)
        targets = torch.full((2, 10), -100, dtype=torch.long)
        
        loss = criterion(logits, targets)
        
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)
        
        loss.backward()
        self.assertIsNotNone(logits.grad)
        # Gradient should be all zeros since all tokens are ignored
        self.assertTrue(torch.all(logits.grad == 0.0))

    def test_focal_loss_reduction_modes(self):
        """
        Verify that reduction='mean' and reduction='sum' function correctly.
        """
        criterion_mean = FocalLoss(gamma=2.0, reduction='mean', ignore_index=-100)
        criterion_sum = FocalLoss(gamma=2.0, reduction='sum', ignore_index=-100)
        
        logits = torch.randn(2, 10, 256, requires_grad=True)
        targets = torch.randint(0, 256, (2, 10), dtype=torch.long)
        
        loss_mean = criterion_mean(logits, targets)
        loss_sum = criterion_sum(logits, targets)
        
        self.assertGreater(loss_sum.item(), loss_mean.item())
        
        total_tokens = 20  # Batch * Seq_Len = 2 * 10
        self.assertTrue(torch.allclose(loss_sum / total_tokens, loss_mean, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
