""" This file is created as the solution template for question 2.3 in DD2434 - Assignment 2.

    Please keep the fixed parameters in the function templates as is (in 2_3.py file).
    However if you need, you can add parameters as default parameters.
    i.e.
    Function template: def calculate_likelihood(tree_topology, theta, beta):
    You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):

    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_3_small_tree, q_2_3_medium_tree, q_2_3_large_tree).
    Each tree have 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.
"""
from collections import defaultdict

import numpy as np
from Tree import Tree
from Tree import Node

tDict = {}  # defaultdict(dict)
sDict = {}  # defaultdict(dict)


def getChildren(nodeIdx, topology):
    children = []
    for i in range(len(topology)):
        if topology[i] == nodeIdx:
            children.append(i)
    return children


def condProbDist(theta, node, val, valParent=None):
    node = int(node)
    val = int(val)
    if valParent is None:
        return theta[node][val]
    else:
        return theta[node][valParent][val]


class NodeVal:
    def __init__(self, nodeIdx, val):
        self.nodeIdx = nodeIdx
        self.val = val

    def __hash__(self):
        return hash(self.nodeIdx) + hash(self.val)

    def __eq__(self, other):
        return self.nodeIdx == other.nodeIdx and self.val == other.val


# likelihood of all nodes beneath node, given nodeVal = val
def s(node, val, topology, theta, beta):
    nodeVal = NodeVal(node, val)
    hashedVal = sDict.get(nodeVal)
    if hashedVal is not None:
        return hashedVal

    children = getChildren(node, topology)
    if len(children) == 0:  # leaf
        valToReturn = 1 if beta[node] == val else 0
        # nodeVal = NodeVal(node, val)
        sDict[nodeVal] = valToReturn
        return valToReturn

    res = [0] * len(children)
    numCategories = len(theta[0])
    for childIdx in range(len(children)):
        for catIdx in range(numCategories):
            res[childIdx] += s(children[childIdx], catIdx, topology, theta, beta) \
                             * condProbDist(theta, children[childIdx], catIdx, val)

    resVal = 1
    for i in range(len(children)):
        resVal *= res[i]

    sDict[nodeVal] = resVal
    return resVal


def getSibling(node, topology):
    for nodeIdx in range(len(topology)):
        if topology[int(nodeIdx)] == topology[int(node)] and node != nodeIdx:
            return nodeIdx

    return -1


def t(node, val, topology, theta, beta):
    nodeVal = NodeVal(node, val)
    hashedVal = tDict.get(nodeVal)
    if hashedVal is not None:
        return hashedVal

    parent = topology[int(node)]
    if np.isnan(parent):  # root
        return condProbDist(theta, node, val)

    sibling = getSibling(node, topology)
    numCategories = len(theta[0])
    res = 0

    if sibling == -1:
        for parentVal in range(numCategories):
            res += condProbDist(theta, node, val, parentVal) \
                   * t(parent, parentVal, topology, theta, beta)

        tDict[nodeVal] = res
        return res

    for parentVal in range(numCategories):
        for siblingVal in range(numCategories):
            res += condProbDist(theta, node, val, parentVal) \
                   * condProbDist(theta, sibling, siblingVal, parentVal) \
                   * s(sibling, siblingVal, topology, theta, beta) \
                   * t(parent, parentVal, topology, theta, beta)

    tDict[nodeVal] = res
    return res


def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    You can change the function signature and add new parameters. Add them as parameters with some default values.
    i.e.
    Function template: def calculate_likelihood(tree_topology, theta, beta):
    You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):
    """

    # TODO Add your code here
    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    print("Calculating the likelihood...")
    # likelihood = np.random.rand()

    for nodeIdx in range(1, len(tree_topology)):
        tree_topology[nodeIdx] = int(tree_topology[nodeIdx])

    '''
    totalLikelihood = 1
    for nodeIdx in range(len(tree_topology)):
        if not np.isnan(beta[nodeIdx]):  # leaf
            leafLikelihood = t(nodeIdx, beta[nodeIdx], tree_topology, theta, beta)
            totalLikelihood += np.log(leafLikelihood)
            print("Likelihood for node:", nodeIdx, ":", leafLikelihood)
    '''
    firstLeaf = -1
    for nodeIdx in range(len(tree_topology)):
        if not np.isnan(beta[nodeIdx]):  # leaf
            firstLeaf = nodeIdx
            break

    totalLikelihood = t(firstLeaf, beta[firstLeaf], tree_topology, theta, beta)

    print("Total likelihood:", totalLikelihood)
    # print("Total likelihood:", np.exp(totalLikelihood))
    # End: Example Code Segment

    return totalLikelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.3.")

    print("\n1. Load tree data from file and print it\n")

    # filename = "data/q2_3_small_tree.pkl"  # "data/q2_3_medium_tree.pkl", "data/q2_3_large_tree.pkl"
    filename = "data/q2_3_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()

    print("\n2. Calculate likelihood "
          "of each FILTERED sample\n")
    # These filtered samples already
    # available in the tree object.
    # Alternatively, if you want, you can
    # load them from corresponding
    # .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        global sDict
        sDict = {}
        global tDict
        tDict = {}
        sample_likelihood = calculate_likelihood(t.get_topology_array(),
                                                 t.get_theta_array(),
                                                 beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
