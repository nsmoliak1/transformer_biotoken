import os
import csv
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# sys.path.append(current_dir)
sys.path.append(parent_dir)


def read_data(filepath):
    with open(os.path.join(current_dir, filepath), mode="r") as file:
        csv_reader = csv.reader(file)
        data = [row for row in csv_reader]

    return data


def remove_best_value(data_array, filepath):
    with open(os.path.join(current_dir, filepath), mode="a") as file:
        csv_writer = csv.writer(file)

        for data in data_array:
            data: list
            indices = list(filter(lambda x: data[x] == "0", range(len(data))))

            new_data = [data[0]]
            new_data.append("<SOS>")
            new_data = new_data + data[1 : indices[0]]
            new_data.append("<NOS>")
            new_data = new_data + data[indices[1] + 1 : indices[2]]
            new_data.append("<EOS>")

            csv_writer.writerow(new_data)


def remove_approximate_value(data_array, filepath):
    with open(os.path.join(current_dir, filepath), mode="a") as file:
        csv_writer = csv.writer(file)

        for data in data_array:
            data: list
            indices = list(filter(lambda x: data[x] == "0", range(len(data))))

            new_data = [data[0]]
            new_data.append("<SOS>")
            new_data = new_data + data[2 : indices[0] - 1]
            new_data = new_data + data[indices[0] + 1 : indices[1]]
            new_data.append("<NOS>")
            new_data = new_data + data[indices[1] + 1 : indices[2]]
            new_data = new_data + data[indices[2] + 1 : indices[3]]
            new_data.append("<EOS>")

            csv_writer.writerow(new_data)


if __name__ == "__main__":
    data_array = read_data("aa_data.csv")
    # remove_best_value(data_array, "new_test_data.csv")
    remove_approximate_value(data_array, "new_test_data.csv")
