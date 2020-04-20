import numpy as np
import sys
from gui import *

def euclidean_distance(equation):
    return np.sum(equation**2)

def formula_to_list(formula,druglist):
    index_list =[]
    for item in formula:
        index_list.append(druglist.index(item))
    return index_list

def list_to_formular(list,druglist):
    formula = ""
    for drug_index in list:
        formula += (druglist[drug_index] +", ")
    return formula[0:len(formula)-2]

def finding_formular(input_list, allergy_list, druglist, limit):
    # getting embedded vectors from input file
    filename = "DS2.npy"
    dataset = np.load(filename)
    output_list ={}

    #find the combined vector of the input list
    input_vector = np.zeros(len(dataset[0]))
    for drug in formula_to_list(input_list, druglist):
        input_vector += dataset[drug]

    #compute list for searching drug
    drug_searching_list =[x for x in range(len(dataset))]

    for allergy_drug in formula_to_list(allergy_list, druglist):
        drug_searching_list.remove(allergy_drug)

    #find one drug combination
    for drug_index in drug_searching_list:
        distance = euclidean_distance(input_vector - dataset[drug_index])
        if distance <= limit:
            output_list[list_to_formular([drug_index], druglist)] = distance

    #find two drug combinations
    for drug_index_i in drug_searching_list:
        for drug_index_j in drug_searching_list:
            if drug_index_i == drug_index_j:
                break
            distance = euclidean_distance(input_vector - dataset[drug_index_i] - dataset[drug_index_j])
            if distance <= limit:
                output_list[list_to_formular([drug_index_j, drug_index_i], druglist)] = distance

    # #find three drug combinations
    # for drug_index_i in drug_searching_list:
    #     for drug_index_j in drug_searching_list:
    #         if drug_index_i == drug_index_j:
    #             break
    #         for drug_index_k in drug_searching_list:
    #             if drug_index_j == drug_index_k:
    #                 break
    #
    #             distance = euclidean_distance(
    #                 input_vector - dataset[drug_index_i] - dataset[drug_index_j] - dataset[drug_index_k])
    #             if distance <= limit:
    #                 output_list[list_to_formular([drug_index_k, drug_index_j, drug_index_i])] = distance

    return output_list

if __name__ == "__main__":


    #getting drug list form first line of drug matrix
    drug_file ="data\DS2\ddiMatrix.csv"
    f = open(drug_file, "r")
    druglist =f.readline().strip().strip(",").split(",")

    #Open the GUI
    root = tk.Tk()
    Application(root, druglist, finding_formular)


    # #For Testing
    # for i in druglist:
    #     result = finding_formular([i],
    #                               [i],
    #                               druglist,
    #                               0.05)
    #     if result != {}:
    #         print(i, end=" ")
    #         for key, item in result.items():
    #             print(key, item)


