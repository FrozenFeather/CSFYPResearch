import numpy as np
from gui import *
class Node:
    def __init__(self, name, height, distance, child_list):
        self.name = name
        self.height = height
        self.distance = distance
        self.child_list = child_list
        self.child_node_list = []

class Tree:
    def __init__(self,root, max_height, limit, max_distance, target_vector):
        self.root = root
        self.max_height = max_height
        self.limit = limit
        self.max_distance = max_distance
        self.target_vector = target_vector
        self.output_list = {}
    def create_tree(self, root):
        child_height = root.height + 1
        for child in root.child_list:
            child_name = root.name.copy()
            child_name.append(child)

            sum_distance = np.zeros(np.shape(dataset[0]))

            for c in child_name:
                sum_distance += dataset[c]

            child_distance = euclidean_distance(sum_distance - self.target_vector)

            if child_distance - self.limit > self.max_distance * (self.max_height - child_height + 1):
                return

            if child_distance <= self.limit:
                self.output_list[list_to_formula([x for x in child_name], druglist)] = child_distance

            if child_height == self.max_height:
                return

            child_index = root.child_list.index(child)
            child_list = root.child_list[child_index+1:]

            tmp_child_node = Node(child_name, child_height,  child_distance, child_list)
            root.child_node_list.append(tmp_child_node)
            self.create_tree(tmp_child_node)

def euclidean_distance(equation):
    return np.sum(equation**2)

def formula_to_list(formula,druglist):
    index_list =[]
    for item in formula:
        index_list.append(druglist.index(item))
    return index_list

def list_to_formula(list,druglist):
    formula = ""
    for drug_index in list:
        formula += (druglist[drug_index] +", ")
    return formula[0:len(formula)-2]

def finding_formula(list,druglist):
    formula = ""
    for drug_index in list:
        formula += (druglist[drug_index] +", ")
    return formula[0:len(formula)-2]


# all the calculation is in here
def finding_formula(input_list, allergy_list, druglist, limit, max_height):

    #find the combined vector of the input list
    input_vector = np.zeros(len(dataset[0]))
    for drug in formula_to_list(input_list, druglist):
        input_vector += dataset[drug]

    #compute list for searching drug
    drug_searching_list =[x for x in range(len(dataset))]

    for allergy_drug in formula_to_list(allergy_list, druglist):
        drug_searching_list.remove(allergy_drug)

    max_distance = 0
    #find the maximum distance of one drug combination
    for drug_index in drug_searching_list:
        distance = euclidean_distance(input_vector - dataset[drug_index])
        if distance > max_distance:
            max_distance = distance
    print(max_distance)
    aNode = Node([], 0, 0, drug_searching_list)
    aTree = Tree(aNode, max_height, limit, max_distance, input_vector)
    aTree.create_tree(aNode)

    return aTree.output_list

if __name__ == "__main__":
    #getting drug list form first line of drug matrix
    drug_file ="data\DS2\ddiMatrix.csv"
    f = open(drug_file, "r")
    druglist =f.readline().strip().strip(",").split(",")

    # getting embedded vectors from input file
    filename = "DS2.npy"
    dataset = np.load(filename)

    #Open the GUI
    root = tk.Tk()
    Application(root, druglist, finding_formula)


    # #For Testing
    # for i in druglist:
    #     result = finding_formula([i],
    #                               [i],
    #                               druglist,
    #                               0.05)
    #     if result != {}:
    #         print(i, end=" ")
    #         for key, item in result.items():
    #             print(key, item)


