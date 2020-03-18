# The input equation should be only consist of drug_id and "+" sign.
# e.g: "1", "1+2", "1 + 2 + 3"

import numpy as np
import sys

def euclidean_distance(equation):
    return np.sum(equation**2)

def result_equation(equation, dataset, max_distance = None):
    drug_list = equation.replace("=", "").replace(" ", "").split("+")

    drug_sum = np.zeros(len(dataset[0]))
    for drug in drug_list:
        drug_sum += dataset[int(drug)]
    tmp_list = [euclidean_distance(drug_sum - dataset[x]) for x in range(len(dataset))]

    # neglect some of the data point
    for drug in drug_list:
        tmp_list[int(drug)] = sys.maxsize

    drug_result_distance = min(tmp_list)

    if max_distance is not None and drug_result_distance > max_distance:
        print("There is no drugs fulfill the equation")
        return None

    return tmp_list.index(drug_result_distance), drug_result_distance


if __name__ == "__main__":
    filename = "9.npy"
    vector = np.load(filename)
    input_equation = input("Input equation:")

    max_distance = input("Input max distance:[Press ENTER to skip] ")
    if max_distance is "":
        max_distance = None
    else:
        max_distance = float(max_distance)

    result = result_equation(input_equation, vector, max_distance)
    if result is not None:
        print(result)

