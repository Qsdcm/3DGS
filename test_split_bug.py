
import torch
import torch.nn as nn
import math

def quaternion_to_rotation_matrix(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros(len(q), 3, 3, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def test_splitting_logic():
    print("Testing Splitting Logic...")
    
    # Setup a single Gaussian
    # Position at origin
    pos = torch.zeros(1, 3)
    
    # Scale: Long on X axis (local)
    scales = torch.tensor([[2.0, 0.5, 0.5]])
    
    # Rotation: 90 degrees around Z axis. 
    # This means the local X axis (long axis) should point along World Y.
    # Quaternion for 90 deg around Z: w=cos(45), z=sin(45) -> w=0.707, z=0.707
    rot = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]])
    
    # Current implementation logic (reproduced)
    longest_axis = torch.argmax(scales).item() # Should be 0 (X axis)
    print(f"Longest axis index: {longest_axis}")
    
    offset = torch.zeros(3)
    offset[longest_axis] = scales[0, longest_axis] * 0.5
    print(f"Offset vector (current logic): {offset}")
    
    new_pos_1 = pos + offset
    print(f"New Position 1 (current logic): {new_pos_1}")
    
    # Expected behavior
    # The Gaussian is rotated 90 deg around Z.
    # Its long axis (Local X) is now pointing along World Y.
    # So we expect the split to happen along World Y.
    # But the current logic adds offset [1.0, 0.0, 0.0] (World X).
    # This splits it along the short axis in world space!
    
    R = quaternion_to_rotation_matrix(rot)
    print(f"Rotation Matrix:\n{R[0]}")
    
    # Correct logic
    offset_local = torch.zeros(3)
    offset_local[longest_axis] = scales[0, longest_axis] * 0.5
    offset_world = torch.matmul(R[0], offset_local)
    
    print(f"Correct Offset World: {offset_world}")
    print(f"Correct New Position 1: {pos + offset_world}")

if __name__ == "__main__":
    test_splitting_logic()
