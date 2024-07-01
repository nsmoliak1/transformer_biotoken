import csv
from enum import Enum, auto
import random
import numpy as np
import generator

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# sys.path.append(current_dir)
sys.path.append(parent_dir)

gen = generator.Generator(
    struc_file=os.path.join(current_dir, "structure.yaml"),
    set_file=os.path.join(current_dir, "sets.yaml"),
    cap_file=os.path.join(current_dir, "cap.yaml"),
)

aa_dict = {
    1: "A",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "O",
    14: "P",
    15: "Q",
    16: "R",
    17: "S",
    18: "T",
    19: "U",
    20: "V",
    21: "W",
    22: "Y",
}


class Category(Enum):
    POT = auto()
    CAP = auto()


def effective_num(num, sig_figs) -> int:
    if num == 0:
        return 0

    order = int(np.floor(np.log10(np.abs(num))))
    effective_float = np.round(num, sig_figs - order - 1)
    return int(effective_float * np.power(10, np.abs(order) + sig_figs - 1))


def get_zeros(curve: np.ndarray) -> np.ndarray:
    zero_points = np.where(
        np.logical_and(np.abs(np.diff(curve)) >= 1e-10, np.diff(np.sign(curve)) != 0)
    )[0]

    return zero_points


def process_curve(curve: np.ndarray, ctg: Category):
    ph_array = np.linspace(0, 14, 1400)

    zero_points = get_zeros(curve)
    extreme = np.array(
        [0, effective_num(curve.min(), 3), effective_num(curve.max(), 3), 0]
    )

    if ctg == Category.POT:
        # add the extend points that gradient is 0.00075
        temp_grad = np.gradient(curve, ph_array[1] - ph_array[0])
        extend_zeros = get_zeros(np.abs(temp_grad) - 0.00075)
        zero_points = np.append(zero_points, (extend_zeros[0], extend_zeros[-1]))
        zero_points.sort()

    return np.append(zero_points, extreme)


def get_token(chara_curves: tuple):
    pot_token = process_curve(chara_curves[0], ctg=Category.POT)
    cap_token = process_curve(chara_curves[1], ctg=Category.CAP)

    return np.append(pot_token, cap_token)


def number_sequence(num, base=23):
    result = list()
    while num > 0:
        temp = num % base
        if temp == 0:
            temp = np.random.randint(1, base)
        result.append(temp)
        num = num // base

    result = [aa_dict[x] for x in result]
    return "".join(result)


def generate_dataset(num, length: int):
    rand_num = np.random.choice(
        np.arange(23 ** (length - 1), 23**length - 1), num, replace=False
    )
    with open(os.path.join(current_dir, "aa_data.csv"), mode="a") as file:
        file_writer = csv.writer(file)
        for num in rand_num:
            aa_chain = number_sequence(num)
            output_list = [aa_chain]
            output_list = output_list + get_token(gen.get(aa_chain)).tolist()
            print(output_list)
            file_writer.writerow(output_list)
    return rand_num


def shuffle_file(filepath):
    with open(os.path.join(current_dir, filepath), mode="r") as in_file:
        lines = in_file.readlines()

    random.shuffle(lines)

    with open(os.path.join(current_dir, filepath), mode="w") as out_file:
        out_file.writelines(lines)


if __name__ == "__main__":
    # result = get_token(gen.get("DE"))
    # print(result)

    shuffle_file("aa_data.csv")

    # generate_dataset(400, 2)
    # generate_dataset(6000, 3)
    # generate_dataset(10000, 4)
