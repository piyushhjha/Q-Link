import numpy as np
import matplotlib.pyplot as plt
from superdense import ChannelParams, estimate_qber
import classical

p_values = np.linspace(0.0, 0.5, 11)
shots = 4096
seed = 42

q_q = []   # quantum (SDC) simulated
q_c = []   # classical (bit-flip) simulated
q_c_formula = []  # classical closed-form 2p - p^2

for p in p_values:
    # Quantum SDC
    qber_q, _ = estimate_qber(ChannelParams(p_depol=p), shots=shots, seed=seed)
    q_q.append(qber_q)

    # Classical (simulate)
    qber_c, _ = classical.simulate_classical(p, shots=shots, seed=seed)
    q_c.append(qber_c)

    # Classical (closed form)
    q_c_formula.append(classical.qber_closed_form(p))

plt.figure(figsize=(7,4.5))
plt.plot(p_values, q_q, marker="o", label="Quantum SDC (sim)")
plt.plot(p_values, q_c, marker="s", linestyle="--", label="Classical 2-bit (sim)")
plt.plot(p_values, q_c_formula, linestyle=":", label="Classical 2-bit (2p - p^2)")
plt.axhline(0.75, linestyle="--", label="Random guess limit (75%)")

plt.title("Message Error (QBER) vs Noise (p)")
plt.xlabel("Noise probability p")
plt.ylabel("QBER (message error)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save for slides; also show
plt.savefig("qber_compare.png", dpi=200)
plt.show()
