
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error


BITS = ["00", "01", "10", "11"]
BIT_TO_PAULI = {
    "00": [],            # I
    "01": ["x"],         # X
    "10": ["z"],         # Z
    "11": ["x", "z"],    # XZ
}


def make_superdense_circuit(bits: str, include_channel_identity: bool = True) -> QuantumCircuit:
    """
    Construct the standard superdense coding circuit for a given 2-bit message.
    Qubit 0 starts with Alice, qubit 1 with Bob. After encoding, Alice's qubit is 'sent' to Bob.
    We model the channel by inserting an identity on qubit 0 that can carry noise in a NoiseModel.

    Returns a circuit that measures both qubits at the end.
    """
    if bits not in BIT_TO_PAULI:
        raise ValueError("bits must be one of '00','01','10','11'")
    qc = QuantumCircuit(2, 2, name=f"SDC_{bits}")

    # 1) Prepare Bell state |Φ+> = (|00> + |11>)/sqrt(2)
    qc.h(0)
    qc.cx(0, 1)

    # 2) Alice encodes two classical bits on her qubit (q0)
    for op in BIT_TO_PAULI[bits]:
        getattr(qc, op)(0)  # apply x/z on q0

    # 3) Channel placeholder: identity on q0 so we can attach noise to 'id'
    if include_channel_identity:
        qc.id(0)

    # 4) Bob decodes
    qc.cx(0, 1)
    qc.h(0)

    # 5) Measure
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


@dataclass
class ChannelParams:
    """Simple container for channel / device parameters."""
    p_depol: float = 0.0           # depolarizing probability on the channel (Alice's qubit)
    t1: float = 90e3               # T1 in nanoseconds (example value)
    t2: float = 110e3              # T2 in nanoseconds (example value)
    gate_time_ns: float = 50.0     # effective time for the 'id' / transfer step


def build_noise_model(params: ChannelParams) -> NoiseModel:
    nm = NoiseModel()

    # Clear channel noise (pick one model at a time to avoid stacking)
    if params.p_depol > 0:
        nm.add_quantum_error(depolarizing_error(params.p_depol, 1), "id", [0])

    elif params.t1 > 0 and params.t2 > 0 and params.gate_time_ns > 0:
        tr = thermal_relaxation_error(params.t1, params.t2, params.gate_time_ns)
        nm.add_quantum_error(tr, "id", [0])

    return nm

def run_message(bits: str, noise: ChannelParams, shots: int = 4096, seed: int | None = 1234) -> Dict[str, int]:
    qc = make_superdense_circuit(bits, include_channel_identity=True)
    nm = build_noise_model(noise)

    # Force AerSimulator to keep the 'id' gate with noise
    backend = AerSimulator(noise_model=nm, basis_gates=nm.basis_gates, seed_simulator=seed)

    # Transpile with the backend basis gates so 'id' survives
    tqc = transpile(qc, backend, basis_gates=nm.basis_gates, optimization_level=0)

    result = backend.run(tqc, shots=shots).result()
    return result.get_counts()



def estimate_qber(noise: ChannelParams, shots: int = 4096, seed: int | None = 1234) -> Tuple[float, np.ndarray]:
    """
    Compute QBER across the 4 messages, and return (qber, confusion_matrix).
    Confusion matrix C[i, j] = P(decode=j | sent=i), in the order BITS.
    """
    cm = np.zeros((4, 4), dtype=float)

    for i, bits in enumerate(BITS):
        counts = run_message(bits, noise, shots=shots, seed=seed)
        total = sum(counts.values())
        for j, out in enumerate(BITS):       
    # Qiskit prints bitstrings with c1 first, then c0
            flipped = out[::-1]  
            cm[i, j] = counts.get(flipped, 0) / total if total else 0.0


    # QBER = 1 - average correct decode probability
    acc = np.mean(np.diag(cm))
    qber = 1.0 - acc
    return qber, cm


def sweep_depolarizing(ps: List[float], shots: int = 4096, seed: int | None = 1234, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep a list of depolarizing probabilities. Returns (ps, qbers).
    """
    qbers = []
    for p in ps:
        noise = ChannelParams(p_depol=float(p), **kwargs)
        qber, _ = estimate_qber(noise, shots=shots, seed=seed)
        qbers.append(qber)
    return np.array(ps, dtype=float), np.array(qbers, dtype=float)
