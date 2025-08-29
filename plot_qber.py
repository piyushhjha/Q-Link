import matplotlib.pyplot as plt
import numpy as np
from superdense import ChannelParams, estimate_qber

# Sweep values of depolarizing probability
p_values = np.linspace(0, 0.5, 11)   # 0.0 to 0.5 in 0.05 steps
qber_values = []

for p in p_values:
    qber, _ = estimate_qber(ChannelParams(p_depol=p), shots=4096, seed=42)
    qber_values.append(qber)

# Plot
plt.figure(figsize=(6,4))
plt.plot(p_values, qber_values, marker='o', linestyle='-', label="Simulated QBER")
plt.axhline(0.75, color="red", linestyle="--", label="Random guess baseline (75%)")

plt.title("QBER vs Depolarizing Noise Probability")
plt.xlabel("Depolarizing Probability (p)")
plt.ylabel("Quantum Bit Error Rate (QBER)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
