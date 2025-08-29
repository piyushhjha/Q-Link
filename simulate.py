
import argparse
import numpy as np

from superdense import sweep_depolarizing, ChannelParams, estimate_qber, BITS


def main():
    parser = argparse.ArgumentParser(description="Q-Link: Superdense coding sweep")
    parser.add_argument("--p", nargs="+", type=float, default=[0.0, 0.02, 0.04, 0.06, 0.08],
                        help="Depolarizing probabilities to test (space-separated)")
    parser.add_argument("--shots", type=int, default=4096, help="Shots per message")
    parser.add_argument("--seed", type=int, default=1234, help="Simulator seed")
    args = parser.parse_args()

    ps, qbers = sweep_depolarizing(args.p, shots=args.shots, seed=args.seed)
    print("Noise sweep (p_depol) vs QBER")
    for p, q in zip(ps, qbers):
        print(f"  p={p:.3f} -> QBER={q*100:.2f}%")

    # Also print a confusion matrix for a representative noise level (use last p)
    representative = float(ps[-1])
    qber, cm = estimate_qber(ChannelParams(p_depol=representative), shots=args.shots, seed=args.seed)
    print("\nRepresentative confusion matrix (rows=sent, cols=decoded) at p={:.3f}".format(representative))
    header = "       " + "  ".join(BITS)
    print(header)
    for i, row in enumerate(cm):
        print(f"{BITS[i]}  " + "  ".join(f"{v:0.2f}" for v in row))
    print(f"\nQBER at p={representative:.3f}: {qber*100:.2f}%")


if __name__ == "__main__":
    main()
