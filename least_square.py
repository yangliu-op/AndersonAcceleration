import numpy as np
# import numpy.random as rand
from derivativetest import derivativetest
from logistic import logistic
from regularizer import regConvex, regNonconvex
# from scipy.sparse import spdiags, csr_matrix
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#global mydtype
#mydtype = torch.float64

def least_square(X, y, w, HProp=1, arg=None, reg=None, act='logistic'):
    """
    Least square problem sum(phi(Xw) - y)^2, where phi is logistic function.
    INPUT:
        X: data matrix
        y: lable vector
        w: variables
        HProp: subsampled(perturbed) Hessian proportion
        arg: output control
        reg: regularizer control
        act: activation function
    OUTPUT:
        f, gradient, Hessian-vector product/Gauss_Newton_matrix-vector product
    """
    if reg == None:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    else:
        reg_f, reg_g, reg_Hv = reg(w, arg=None)
        
    n, d = X.shape
    # X = csr_matrix(X)    
    if act == 'logistic':
        fx, grad, Hess = logistic(X, w)
        
    #output control with least computation
    
    if arg == 'diff':     
        def diffcalc(X, fx, v):
            fxn, _, _ = logistic(X, v)
            return torch.sum((fxn+fx-2*y)*(fxn-fx))/n 
        diff = lambda t: diffcalc(X, fx, t) + reg(w, arg='rdiff')(t)
        return diff  
    
    
    f = torch.sum((fx-y)**2)/n + reg_f
    if arg == 'f':        
        return f.detach()   
    
    
    g = 2*torch.mv(X.T, grad*(fx-y))/n + reg_g
        
    if arg == 'g':        
        return g.detach()
        
    if arg == 'fg':        
        return f.detach(), g.detach()
    
    if arg == 'H':
        #W is NxN diagonal matrix of weights with ith element=s2
        Hess = (2*(grad**2 + Hess*(fx-y))/n)
        H = ((X.T * Hess) @ X).detach()
        return -torch.lobpcg(-H, k=1, largest=True, tol=1e-4)[0]
        
    
    if arg == None:
        if HProp == 1:
            #W is NxN diagonal matrix of weights with ith element=s2
            Hess = 2*(grad**2 + Hess*(fx-y))/n
            Hv = lambda v: hessvec(X, Hess, v) + reg_Hv(v)
            return f, g, Hv
        else:
            n_H = np.int(np.floor(n*HProp))
            idx_H = np.random.choice(n, n_H, replace = False)
            if act == 'logistic':
                fx_H, grad_H, Hess_H = logistic(X[idx_H,:], w)       
            Hess = 2*(grad_H**2 + Hess_H*(fx_H-y[idx_H,:]))/n
            Hv = lambda v: hessvec(X[idx_H,:], Hess, v) + reg_Hv(v)
            return f.detach(), g.detach(), Hv
    
    if arg == 'gn': #hv product        
        if HProp == 1:
            Hess_gn = 2*grad**2/n
            Hv = lambda v: (hessvec(X, Hess_gn, v) + reg_Hv(v)).detach()
            return f, g, Hv    
        else:
            n_H = np.int(np.floor(n*HProp))
            idx_H = np.random.choice(n, n_H, replace = False)
            if act == 'logistic':
                fx_H, grad_H, Hess_H = logistic(X[idx_H,:], w)          
            Hess_gn = 2*grad_H**2/n_H
            Hv = lambda v: (hessvec(X[idx_H,:], Hess_gn, v) + reg_Hv(v)).detach()
            return f.detach(), g.detach(), Hv
    

def hessvec(X, Hess, v):
    Xv = torch.mv(X, v)
    Hv = torch.mv(X.T, Hess*Xv)
    return Hv

#@profile
def main():        
    import torch.utils.data as data
    import torchvision.datasets as datasets
    from  torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    data_dir = '../Data'
    train_Set = datasets.MNIST(data_dir, train=True,
                                transform=transforms.ToTensor())
    (n, d, d) = train_Set.data.shape
    d = d**2
    X = train_Set.data.reshape(n, d)
    X = X/255
    X = X.double()
    Y_index = train_Set.targets
    Y = (Y_index>5)*torch.tensor(1).double()
    
#    print(Y)
    lamda = 1
    w = torch.randn(d, dtype=torch.float64)
    # reg = None
    reg = lambda x, arg=None: regConvex(x, lamda, arg)
    # reg = lambda x: regNonconvex(x, lamda)
    fun1 = lambda x, arg=None: least_square(X,Y,x,act='logistic', HProp=1, arg=arg,reg = reg)
    record_txt = lambda filename, myrecord: np.savetxt(
            os.path.join('', filename+'.txt'), myrecord.cpu(), delimiter=',') 
    HH = fun1(w, 'H')
    record_txt('xx', w)
    print(fun1(w, 'diff')(2*w)- fun1(2*w, 'f') + fun1(w,'f'))
    # print()
    ww = np.loadtxt(open('xx.txt',"rb"),delimiter=",",skiprows=0)
    HH = fun1(w, 'H')
    derivativetest(fun1,w)    
#    
if __name__ == '__main__':
    main()