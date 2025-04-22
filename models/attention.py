import sys
import torch
import torch.nn as nn
from ultralytics.nn.tasks import parse_model  # noqa: E402 – after torch import

class CBAM(nn.Module):
    """Channel + Spatial Attention (with residual)"""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.expected_channels = channels
        # We'll initialize the actual layers in the first forward pass
        self.initialized = False
        # Placeholders
        self.ca = None
        self.sa = None
        self.actual_channels = None
        
    def _initialize(self, x):
        """Initialize layers based on actual input tensor dimensions"""
        channels = x.shape[1]
        self.actual_channels = channels
        r = max(channels // 16, 4)  # Reduction ratio, minimum 4
        
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.initialized = True
        
        # Print diagnostic info
        if self.expected_channels != channels:
            print(f"CBAM warning: Expected {self.expected_channels} channels but got {channels}.")

    def forward(self, x):  # type: ignore
        # Initialize on first forward pass or if channel count changed
        if not self.initialized or x.shape[1] != self.actual_channels:
            self._initialize(x)
            
        identity = x
        # Channel
        x = x * self.ca(x)
        # Spatial
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        x = x * self.sa(torch.cat([avg, mx], dim=1))
        return x + identity  


def register_attention():
    for m in [
        'ultralytics.nn.modules',
        'ultralytics.nn.modules.block',
        'ultralytics.nn.tasks',
        '__main__',
    ]:
        if m in sys.modules:
            setattr(sys.modules[m], 'CBAM', CBAM)
    
    # parse_model 在 runtime 内部用到 globals()
    parse_model.__globals__['CBAM'] = CBAM
    
    # Register in the current module too
    globals()['CBAM'] = CBAM

if __name__ == "__main__":
    register_attention()
    # Test with different channel counts
    for ch in [16, 32, 64, 128]:
        dummy = torch.randn(1, ch, 80, 80)
        net = CBAM(64)  # Still works even though we specify 64
        out = net(dummy)
        print(f"Input: {dummy.shape} -> Output: {out.shape}")