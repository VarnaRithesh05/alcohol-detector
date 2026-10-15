"""
nwd_loss.py — Normalized Wasserstein Distance (NWD) Loss Formulation

Implements the continuous distribution loss for micro-scale objects (<32x32 px)
as defined in research on tiny object detection:
1. Models bounding boxes A=(cx_a, cy_a, w_a, h_a) and B=(cx_b, cy_b, w_b, h_b) as 2D Gaussian distributions.
2. Computes the 2nd Wasserstein Distance:
   W_2^2(N_a, N_b) = ||[cx_a, cy_a] - [cx_b, cy_b]||_2^2 + ((w_a - w_b)/2)^2 + ((h_a - h_b)/2)^2
3. Normalizes into scale-invariant metric:
   NWD(A, B) = exp( - sqrt(W_2^2(N_a, N_b)) / C )
   where C is a tuned constant matching target resolution (e.g. C = 12.8 for 640x640 images).
4. Loss: L_NWD = 1 - NWD(A, B)
"""

import torch
import torch.nn as nn
import numpy as np

def compute_nwd(bboxes1: torch.Tensor, bboxes2: torch.Tensor, constant: float = 12.8, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Normalized Wasserstein Distance between two sets of bounding boxes.
    Boxes formatted as xyxy: [x1, y1, x2, y2]
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2.chunk(4, -1)

    # Centers
    cx1, cy1 = (b1_x1 + b1_x2) / 2.0, (b1_y1 + b1_y2) / 2.0
    cx2, cy2 = (b2_x1 + b2_x2) / 2.0, (b2_y1 + b2_y2) / 2.0

    # Width and Height
    w1, h1 = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)

    # 2nd Wasserstein Distance squared:
    # W_2^2 = (cx1 - cx2)^2 + (cy1 - cy2)^2 + ((w1 - w2)/2)^2 + ((h1 - h2)/2)^2
    center_dist_sq = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)
    shape_dist_sq = ((w1 - w2) / 2.0).pow(2) + ((h1 - h2) / 2.0).pow(2)
    wasserstein_dist = torch.sqrt((center_dist_sq + shape_dist_sq).clamp(min=eps))

    # Normalized Wasserstein Distance
    nwd = torch.exp(-wasserstein_dist / constant)
    return nwd

class NWDLoss(nn.Module):
    """
    Normalized Wasserstein Distance Loss module for YOLOv8 regression replacement/augmentation.
    """
    def __init__(self, constant: float = 12.8):
        super().__init__()
        self.constant = constant

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        pred_boxes: Tensor of shape (..., 4) [x1, y1, x2, y2]
        target_boxes: Tensor of shape (..., 4) [x1, y1, x2, y2]
        """
        nwd = compute_nwd(pred_boxes, target_boxes, constant=self.constant)
        loss = 1.0 - nwd
        return loss.mean()

if __name__ == "__main__":
    # Test verification
    box_gt = torch.tensor([[100.0, 100.0, 120.0, 140.0]]) # 20x40 micro target
    box_pred_close = torch.tensor([[101.0, 102.0, 121.0, 142.0]]) # 1-2px jitter
    box_pred_far = torch.tensor([[110.0, 115.0, 130.0, 155.0]])

    nwd_fn = NWDLoss(constant=12.8)
    loss_close = nwd_fn(box_pred_close, box_gt).item()
    loss_far = nwd_fn(box_pred_far, box_gt).item()

    print(f"NWD Loss (Close prediction): {loss_close:.4f}")
    print(f"NWD Loss (Far prediction):   {loss_far:.4f}")
    assert loss_close < loss_far, "NWD loss must scale monotonically with distance!"
    print("NWD Loss module operational and verified.")
