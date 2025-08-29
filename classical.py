import numpy as np

# Map between indices 0..3 <-> bit pairs (00,01,10,11)
IDX2BITS = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=int)

def simulate_classical(p: float, shots: int = 4096, seed: int | None = 42):
    rng = np.random.default_rng(seed)

    # Uniformly sample messages 0..3 (representing 00,01,10,11)
    sent_idx = rng.integers(0, 4, size=shots)
    sent_bits = IDX2BITS[sent_idx]                      # shape (shots, 2)

    # Apply independent bit flips to each bit with prob p
    flips = rng.random((shots, 2)) < p
    recv_bits = sent_bits ^ flips

    # Convert received bits back to indices 0..3
    recv_idx = recv_bits[:, 0] * 2 + recv_bits[:, 1]

    # Build 4x4 confusion matrix (rows: sent, cols: decoded)
    cm = np.zeros((4, 4), dtype=float)
    for s, r in zip(sent_idx, recv_idx):
        cm[s, r] += 1.0

    # Normalize each row to probabilities
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_sums, where=row_sums != 0)

    # "QBER" here = message error probability (1 - mean correct-diagonal)
    qber = 1.0 - np.mean(np.diag(cm))
    return qber, cm

def qber_closed_form(p: float) -> float:
    """
    Exact message error for two independent bit flips:
    message is correct only if neither bit flips -> (1 - p)^2.
    So error = 1 - (1 - p)**2 = 2p - p**2
    """
    return 2.0 * p - p * p
