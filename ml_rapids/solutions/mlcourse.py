import numpy as np
from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

def f_true(x):
    w = (0.2, 0.01,0.025, 0.1)
    return polyval(x, w)

def Phi(x, k):
    return np.concatenate([x ** ki for ki in range(k+1)], axis=1)

def fit(x, y, k, lam=0):
    x = x.reshape((x.size, 1))
    #y = y.reshape((y.size, 1))
    X = Phi(x, k)
    A = np.dot(X.T, X) 
    if lam != 0:
        if np.isscalar(A):
            A += lam
        else:
            A += lam * np.eye(A.shape[0])
    b = np.dot(X.T, y)
    if np.isscalar(A):
        return b / A
    return np.linalg.solve(A,b)

def predict(x, w):
    x = x.reshape((x.size, 1))
    if np.isscalar(w):
        k = 0
    else:
        k = len(w) - 1
    X = Phi(x, k)
    yhat = np.dot(X, w)
    return yhat


def mse(y, yhat):
    return np.mean((y - yhat)**2)



class Node:
    def __init__(self):
        self.split = None
        self.left = None
        self.right = None
        self.value = None
        
def calc_score(g,h,it):
    gl = sum(g[:it+1])
    gr = sum(g[it+1:])
    hl = sum(h[:it+1])
    hr = sum(h[it+1:])
    return  gl*gl / hl + gr*gr / hr - (gl+gr)**2 / (hl+hr)

def build1d(x, g, h, d, max_depth):
    n = Node()
    n.value = -sum(g) / sum(h) 
    
    if d == max_depth or len(x) == 1: #nothing to do
        return n
    
    #find split pos and split in the middle
    max_score, split = max([(calc_score(g,h,it) , it) for it in range(len(x) - 1)])
    n.split = 0.5*(x[split] + x[split+1])
    
    #recurse on the split
    n.left = build1d(x[:split+1], g[:split+1], h[:split+1], d+1, max_depth)
    n.right = build1d(x[split+1:], g[split+1:], h[split+1:], d+1, max_depth)
    return n

def eval1d(n, x):
    if n.split is None:
        return n.value
    if x < n.split:
        return eval1d(n.left, x)
    return eval1d(n.right, x)

def xgboost1d(xs, ys, rounds, depth):
    forrest = []
    yprev = np.zeros_like(ys)
    h = np.ones(len(ys))
    for r in range(rounds):
        g = yprev - ys
        root = build1d(xs, g, h, 0, depth)
        yprev += np.array([eval1d(root, x_) for x_ in xs])
        forrest.append(root)
    return forrest

def pred1d(forrest, x):
    pred = np.zeros_like(x).flatten()
    for tree in forrest:
        pred += np.array([eval1d(tree, x_) for x_ in x])
    return pred

def plot_region1d(root, x, y):
    plt.figure(figsize=(12,4))
    plt.plot(x,y, label='f(x)')
    #plt.plot(x,y, 'o')
    #plt.plot(xtest,pred_ref, 'o', ms=1)
    #plt.plot(xtest,pred_test, '-', ms=1, label="f(x)")
    plt.xlim([-1,1])
    plt.ylim([0,0.3])
    plt.xlabel("x")
    plt.ylabel("y")

    def shade_rec(node, l, r, d, is_right):
        if not node:
            return
        #shade from l to r
        plt.fill_between([l,r], 0, 0.3 , alpha=1, facecolor='white')
        plt.fill_between([l,r], 0, 0.3 , alpha=0.2)#, facecolor='123456')
        if node.split:
            plt.text(node.split-0.07, node.value, "x < {:.2f}".format(node.split))

        #recurse
        shade_rec(node.left, l, node.split, d+1, False)
        shade_rec(node.right, node.split, r, d+1, True)
        

    shade_rec(root, -1,1, 0, True)
    plt.legend()

        
