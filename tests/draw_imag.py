import numpy as np
from data.generate_dataset import get_generator, get_zeros
import matplotlib.pyplot as plt

gen = get_generator()
ph_array = np.linspace(0, 14, 1400)
base_line = np.zeros(1400)


def compare_sequences(sequences: list):
    style_list = ["-", "--"]
    labels=['pred', 'true']
    for aa_chain, style, label in zip(sequences, style_list, labels):
        print(f"chain is {aa_chain}")
        temp_pot, temp_cap = gen.get(aa_chain)
        plt.plot(ph_array, temp_pot, label=f"{label}:{aa_chain} - pot", linestyle=style)
        plt.plot(ph_array, temp_cap, label=f"{label}:{aa_chain} - cap", linestyle=style)
        # break

    plt.legend()
    plt.savefig("compare.png")


def one_amino(aa_chain):
    pot, cap = gen.get(aa_chain)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(ph_array, pot, label=f"{aa_chain} - potential")
    zero_points = get_zeros(pot)
    ax1.scatter(
        ph_array[zero_points],
        pot[zero_points],
        marker="o",
        c="none",
        edgecolors="r",
        s=200,
        zorder=5,
    )
    max_min_position = [np.argmax(pot), np.argmin(pot)]
    ax1.scatter(
        ph_array[max_min_position],
        pot[max_min_position],
        marker="^",
        edgecolors="r",
        color="r",
        s=100,
        zorder=5,
    )
    ax1.plot(ph_array, base_line, linestyle="--")
    ax1.set_title(f"{aa_chain} - Potential")
    ax1.set_xlabel("pH")
    ax1.set_ylabel("Potential")
    ax1.legend()

    ax2.plot(ph_array, cap, label=f"{aa_chain} - capacitance", color="green")
    zero_points = get_zeros(cap)
    ax2.scatter(
        ph_array[zero_points],
        cap[zero_points],
        marker="o",
        c="none",
        edgecolors="r",
        s=200,
        zorder=5,
    )
    max_min_position = [np.argmax(cap), np.argmin(cap)]
    ax2.scatter(
        ph_array[max_min_position],
        cap[max_min_position],
        marker="^",
        edgecolors="r",
        color="r",
        s=100,
        zorder=5,
    )
    ax2.plot(ph_array, base_line, linestyle="--", color="orange")
    ax2.set_title(f"{aa_chain} - Capacitance")
    ax2.set_xlabel("pH")
    ax2.set_ylabel("Capacitance")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{aa_chain}.svg", transparent=True)
