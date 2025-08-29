
import numpy as np
from superdense import ChannelParams, estimate_qber

def test_noiseless_is_perfect():
    qber, cm = estimate_qber(ChannelParams(p_depol=0.0), shots=4096, seed=42)
    assert qber < 0.01, f"QBER should be ~0 with no noise; got {qber}"

def test_extreme_noise_is_randomish():
    # A very large depolarizing probability should drive accuracy toward 25% (QBER ~ 75%)
    qber, cm = estimate_qber(ChannelParams(p_depol=0.9), shots=8192, seed=42)
    assert 0.60 <= qber <= 0.90, f"Extreme noise should give QBER around 0.75; got {qber:.2f}"
