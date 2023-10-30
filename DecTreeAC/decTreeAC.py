import numpy as np

class Node:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.children = {}
        self.isLeaf = False
        self.splitAttr = None

class DecisionTreeAC:
    def __init__(self):
        self.root = None
        self.attr = None
        self.label = None

    def fit(self, X, y):
        self.attr = X.columns
        self.label = y
        self.root = self.buildTree(X, y)

    def buildTree(self, X, y):
        node = Node(X, y)
        if self.isPure(y):
            node.isLeaf = True
            node.label = y.values[0]
            return node
        else:
            splitAttr = self.get_split_attr(X, y)
            node.splitAttr = splitAttr
            node.children = self.splitData(X, y, splitAttr)
            for k, v in node.children.items():
                node.children[k] = self.buildTree(v.data, v.label)
            return node
    
    def isPure(self, y):
        return len(np.unique(y)) == 1
    
    def get_split_attr(self, X, y):
        splitAttr, maxGain = None, -1e9
        for attr in X.columns:
            gain = self.informationGain(X, y, attr)
            if gain > maxGain:
                splitAttr, maxGain = attr, gain
        return splitAttr
    
    def informationGain(self, X, y, attr):
        gain = self.entropy(y)
        for v in np.unique(X[attr]):
            sub_y = y[X[attr] == v]
            gain -= (len(sub_y) / len(y)) * self.entropy(sub_y)
        return gain
    
    def entropy(self, y):
        entropy = 0
        for v in np.unique(y):
            prob = len(y[y == v]) / len(y)
            entropy -= prob * np.log2(prob)
        return entropy
    
    def splitData(self, X, y, splitAttr):
        children = {}
        for val in np.unique(X[splitAttr]):
            children[val] = Node(X[X[splitAttr] == val].drop([splitAttr], axis=1), y[X[splitAttr] == val])
        return children
    
    def predict(self, X):
        y = []
        for i in range(len(X)):
            y.append(self.predictOne(X.iloc[i]))
        return y
    
    def predictOne(self, x):
        node = self.root
        while not node.isLeaf:
            node = node.children[x[node.splitAttr]]
        return node.label
    
    def printTree(self):
        self.printNode('Root', self.root, 0)

    def printNode(self, parent_key, node, depth):
        if node.isLeaf:
            print((depth-1) * ' | ', f'|-[{parent_key} -> Label {node.label}]')
        else:
            print((depth-1) * ' | ', f'|-({parent_key} -> Attr {node.splitAttr})')
            for k, child in node.children.items():
                self.printNode(k, child, depth + 1)