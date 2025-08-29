import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import classical

from superdense import ChannelParams, estimate_qber, BITS, sweep_depolarizing

st.set_page_config(page_title="Q-Link: Superdense Coding", layout="centered")

st.title("Q-Link: Superdense Coding Simulator")
st.caption("Adjust channel noise and shots to observe decoding reliability (QBER).")

# --- Interactive Widgets ---
col1, col2 = st.columns(2)
with col1:
    p = st.slider("Depolarizing probability (channel)", min_value=0.0, max_value=0.2, step=0.005, value=0.02)
with col2:
    shots = st.selectbox("Shots per message", options=[512, 1024, 2048, 4096, 8192], index=3)

# ✅ NEW: Mode toggle
mode = st.radio(
    "Channel type",
    ["Quantum (Superdense Coding, 1 qubit)", "Classical (Bit-flip, 2 bits)"],
    horizontal=True
)

# --- Analysis & Metrics ---
if mode.startswith("Quantum"):
    noise = ChannelParams(p_depol=float(p))
    qber, cm = estimate_qber(noise, shots=shots)
    note = "Quantum SDC uses 1 qubit per message with prior entanglement."
else:
    qber, cm = classical.simulate_classical(p=float(p), shots=shots, seed=42)
    note = "Classical baseline sends 2 independent bits through a bit-flip channel."

st.metric("QBER", f"{qber*100:.2f}%")

df = pd.DataFrame(cm, index=[f"sent {b}" for b in BITS], columns=[f"decoded {b}" for b in BITS])
st.subheader("Confusion matrix")
st.dataframe(df.style.format("{:.2f}"))
st.caption(note)

# --- Static Graph Generation (Quantum + Classical) ---
st.subheader("QBER vs. Noise (Static)")

noise_values_static = np.arange(0.0, 0.20, 0.01).tolist()
ps_static, qbers_quantum = sweep_depolarizing(noise_values_static, shots=shots)

# Classical closed-form and simulated
qbers_classical_sim = []
for p_val in noise_values_static:
    qber_c, _ = classical.simulate_classical(p_val, shots=shots, seed=42)
    qbers_classical_sim.append(qber_c)
qbers_classical_formula = [classical.qber_closed_form(p) for p in noise_values_static]

fig_static, ax_static = plt.subplots()
ax_static.plot(ps_static, qbers_quantum, 'bo-', label='Quantum (Superdense)')
ax_static.plot(ps_static, qbers_classical_sim, 'gs--', label='Classical (Simulated)')
ax_static.plot(ps_static, qbers_classical_formula, 'r-', label='Classical (Formula)')
ax_static.axhline(y=0.05, color='k', linestyle='--', label='5% Operational Threshold')
ax_static.set_xlabel("Channel Noise Probability")
ax_static.set_ylabel("Error Rate (QBER)")
ax_static.set_title("Q-Link: Quantum vs Classical Reliability")
ax_static.legend()
ax_static.grid(True)
st.pyplot(fig_static)

# --- Animated Graph Generation (Quantum + Classical) ---
st.subheader("Live Channel Simulation")
threshold = 0.05
max_p_animated = st.slider("Maximum noise for live graph", min_value=0.05, max_value=0.20, step=0.01, value=0.1)

# Compute quantum QBER
noise_values_animated = np.arange(0.0, max_p_animated + 0.01, 0.01).tolist()
ps_animated, qbers_quantum_animated = sweep_depolarizing(noise_values_animated, shots=shots)

# Compute classical QBER
qbers_classical_animated = []
for p_val in noise_values_animated:
    qber_c, _ = classical.simulate_classical(p_val, shots=shots, seed=42)
    qbers_classical_animated.append(qber_c)

# ✅ NEW: Spy Attack simulation
spy_attack = st.checkbox("Simulate Spy Attack (Eavesdropping)")
if spy_attack:
    qbers_quantum_animated = [min(1.0, q + 0.2) for q in qbers_quantum_animated]

# --- Plot both curves ---
fig_live, ax_live = plt.subplots()
ax_live.plot(ps_animated, qbers_quantum_animated, 'bo-', label='Quantum (Superdense)')
ax_live.plot(ps_animated, qbers_classical_animated, 'gs--', label='Classical')
ax_live.axhline(y=threshold, color='k', linestyle='--', label='5% Operational Threshold')
ax_live.set_xlabel("Channel Noise Probability")
ax_live.set_ylabel("Error Rate (QBER)")
ax_live.set_title("Live Channel Simulation (Quantum vs Classical)")
ax_live.set_xlim(0, max_p_animated)
ax_live.set_ylim(0, 0.3)
ax_live.grid(True)
ax_live.legend()
st.pyplot(fig_live)

# --- Threshold crossing detection ---
crossed_quantum = next((p for p, q in zip(noise_values_animated, qbers_quantum_animated) if q >= threshold), None)
crossed_classical = next((p for p, q in zip(noise_values_animated, qbers_classical_animated) if q >= threshold), None)

# --- Display results ---
if crossed_classical and crossed_quantum:
    if crossed_classical < crossed_quantum:
        st.warning(f"Classical QBER crossed threshold first at noise={crossed_classical:.2f}")
        st.info(f"Quantum QBER crossed threshold later at noise={crossed_quantum:.2f}")
        st.success(f"Quantum survives ~{crossed_quantum/crossed_classical:.1f}x longer under noise.")
    else:
        st.warning(f"Quantum QBER crossed threshold first at noise={crossed_quantum:.2f}")
        st.info(f"Classical QBER crossed threshold later at noise={crossed_classical:.2f}")
else:
    if crossed_classical:
        st.warning(f"Classical QBER crossed threshold at noise={crossed_classical:.2f}")
        st.success("Quantum stayed below threshold in this range.")
    elif crossed_quantum:
        st.warning(f"Quantum QBER crossed threshold at noise={crossed_quantum:.2f}")
        st.success("Classical stayed below threshold in this range.")
    else:
        st.success("Both Quantum and Classical stayed safe below threshold.")

# ✅ NEW: Defense Scenario Message demo
st.subheader("Defense Scenario: Message Transmission")

def send_message(msg: str, error_rate: float):
    noisy = ""
    rng = np.random.default_rng(42)
    for ch in msg:
        if rng.random() < error_rate:
            noisy += "?"   # corrupted char
        else:
            noisy += ch
    return noisy

message = st.text_input("Enter secret message", "Attack at dawn")
if message:
    noisy_classical = send_message(message, qbers_classical_animated[-1])
    noisy_quantum   = send_message(message, qbers_quantum_animated[-1])
    st.write(f"Classical Transmission: {noisy_classical}")
    st.write(f"Quantum Transmission: {noisy_quantum}")

# ✅ NEW: Export option
if st.button("Export Results to CSV"):
    df_results = pd.DataFrame({
        "Noise": ps_animated,
        "Quantum_QBER": qbers_quantum_animated,
        "Classical_QBER": qbers_classical_animated
    })
    df_results.to_csv("results.csv", index=False)
    st.success("Results exported as results.csv")

