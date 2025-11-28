#!/usr/bin/env python
# coding: utf-8

# In[2]:


#------------------------------
# Hierarchical Classifier
# ------------------------------
import numpy as np
class HCNode:
    def __init__(self, classifier=None, left=None, right=None, label=None,pairs=None):
        self.classifier = classifier
        self.left = left
        self.right = right
        self.label = label
        self.pairs = pairs 

class HierarchicalClassifier:
    def __init__(self, base_model):
        """
        base_model: function(X_train, y_train) -> fitted classifier
        """
        self.base_model = base_model
        self.root = None

    def fit(self, X, y):
        classes = np.unique(y)
        self.root = self._build_tree(X, y, classes)
        return self

    # To build the tree structure of classifier
    def _build_tree(self, X, y, classes):
        print("Building node:")
        print("  classes:", classes)
        print("  X.shape:", X.shape)
        print("  y.shape:", y.shape)
        if len(classes) == 1:
            return HCNode(label=classes[0])
        print(f"Building node: classes={classes}, X.shape={X.shape}, y.shape={y.shape}")
        # The largest class_size vs others
        class_sizes = {c: sum(y == c) for c in classes}
        largest_class = max(class_sizes, key=class_sizes.get)
        
        # binary label
        binary_y = (y == largest_class).astype(int)
        
        clf = self.base_model(X, binary_y)
        pairs = clf.pairs if hasattr(clf, "pairs") else None
        
        #left_classes = [largest_class]
        #right_classes = [c for c in classes if c != largest_class]
        
        #left_mask = np.isin(y, left_classes)
        #right_mask = np.isin(y, right_classes)
        
        left_mask = (y == largest_class)
        right_mask = ~left_mask
    
        X_left, y_left = X[:, left_mask], y[left_mask]
        X_right, y_right = X[:, right_mask], y[right_mask]
  
        
        #left_child = self._build_tree(X[:, left_mask], y[left_mask], left_classes)
        #right_child = self._build_tree(X[:, right_mask], y[right_mask], right_classes) 
        
        #left_child = self._build_tree(X_left, y_left, np.unique(y_left)) if X_left.shape[1] > 0 else HCNode(label=largest_class)
        #right_child = self._build_tree(X_right, y_right, np.unique(y_right)) if X_right.shape[1] > 0 else HCNode(label=right_mask[0] if len(right_mask) > 0 else largest_class)
        left_child = self._build_tree(X_left, y_left, np.unique(y_left)) if X_left.shape[1] > 0 else HCNode(label=largest_class)
        remaining_classes = [c for c in classes if c != largest_class]
        if len(remaining_classes) == 1:
            right_child = HCNode(label=remaining_classes[0])
        else:
            right_child = self._build_tree(X_right, y_right, np.unique(y_right)) if X_right.shape[1] > 0 else HCNode(label=remaining_classes[0])
        return HCNode(classifier=clf, left=left_child, right=right_child, pairs=pairs)
    

    def predict(self, X):
        return np.array([self._predict_single(X[:, k])[0] for k in range(X.shape[1])])

    #def predict(self, X):
       # labels = []
       #for i in range(X.shape[1]):
       #     label, _ = self._predict_single(X[:, i])
       #     labels.append(label)
       # return np.array(labels)

    #def predict_score(self, X):
    #   scores = []
    #  for k in range(X.shape[1]):
    #        _, score = self._decision_single(X[:,k])
    #        scores.append(score)
    #    return np.array(scores)
    
    def predict_score(self, X):
        return np.array([self._predict_single(X[:, k])[1] for k in range(X.shape[1])])


    def _predict_single(self, x):
        node = self.root
        score=0
        while node.label is None:
            if node.classifier is None:
                # if the classifier of node is none then choose the right
                node = node.left if node.left is not None else node.right
                if node is None:
                    raise ValueError("Tree node has no classifier and no children")
                continue
            # x.shape = (features,)
            s= node.classifier.decision_function(x[:, np.newaxis])# for Roc score
            s = float(s)
            score += s
            p = node.classifier.predict(x[:, np.newaxis])[0]
            node = node.left if p == 1 else node.right
        # to give the node.label and score for confusion matrix
        return node.label,score
    
    def get_all_pairs(self):
        result = []
        def traverse(node):
            if node is None:
                return
            if node.pairs is not None:
                result.append(node.pairs)
            traverse(node.left)
            traverse(node.right)
        traverse(self.root)
        return result
    
    def decision_function(self, X):
    # return every gene of score（one array）
        scores = []
        for k in range(X.shape[1]):
            _, score = self._predict_single(X[:, k])
            scores.append(score)
        return np.array(scores)

