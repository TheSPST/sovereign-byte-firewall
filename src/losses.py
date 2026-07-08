import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Numerically stable Focal Loss implementation for next-byte sequence prediction.
    Formulation: FL(p_t) = - (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, reduction='mean', ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs expected: (Batch, Sequence_Length, Vocab_Size)
        # targets expected: (Batch, Sequence_Length)
        
        if inputs.dim() > 2:
            # Flatten to (B * L, C)
            inputs = inputs.reshape(-1, inputs.size(-1))
            targets = targets.reshape(-1)
            
        # Filter out padding tokens (e.g. default ignore_index of -100)
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            inputs = inputs[valid_mask]
            targets = targets[valid_mask]
            
            if inputs.numel() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        log_pt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_pt)
        
        # Gather probabilities for the true target classes
        log_pt = log_pt.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        pt = pt.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        # Calculate modulated loss
        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
