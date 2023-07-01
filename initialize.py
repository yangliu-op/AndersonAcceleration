import numpy as np
import torch
from numpy.linalg import inv, svd
from numpy.random import multivariate_normal, randn, rand
from logistic import logit
from optim_algo import AndersonAcc, L_BFGS
from loaddata import loaddata
from showFigure import showFigure
from sklearn import preprocessing
from student_t import student_t
from softmax import softmax
from logitreg import logitreg
from regularizer import regConvex, regNonconvex
import os
# from Auto_Enconder import AE_Call
from least_square import least_square
from scipy import sparse
#import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
        
def initialize(data, methods, prob, x0Type, algPara, lamda=1, seed=None): 
    """
    data: name of chosen dataset
    methods: name of chosen algorithms
    prob: name of chosen objective problems
    regType: type of regularization 
    x0Type: type of starting point
    algPara: a class that contains:
        mainLoopMaxItrs: maximum iterations for main loop
        funcEvalMax: maximum oracle calls (function evaluations) for algorithms
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        show: print result for every iteration
    lamda: parameter of regularizer
    """
    print('Initialization...')
    prob = prob[0]
    x0Type = x0Type[0]
        
    print('Problem:', prob, end='  ')
    if hasattr(algPara, 'cutData'):
        print('Data-size using: ', algPara.cutData)
    print('regulization = %8s' % algPara.regType, end='  ')
    print('gradTol = %8s' % algPara.gradTol, end='  ')
    print('Starting point = %8s ' % x0Type)  
    algPara.regType = algPara.regType[0]
    #smooth regulizer      
    if algPara.regType == 'None':
        reg = None
    if algPara.regType == 'Convex':
        reg = lambda x, arg=None: regConvex(x, lamda, arg)
    if algPara.regType == 'Nonconvex':
        reg = lambda x, arg=None: regNonconvex(x, lamda, arg)
    if algPara.regType == 'Nonsmooth':
        reg = None
          
    filename = '%s_%s_Orc_%s_x0_%s_reg_%s_m_%s_L_%s_Gamma_%s_c1_%s_c2_%s_c3_%s_nu_%s' % (
            prob, data, algPara.funcEvalMax, x0Type, lamda, algPara.Andersonm, 
            algPara.L, algPara.gamma, algPara.c1, algPara.c2, algPara.c3, algPara.nu) 
        
    mypath = filename
    print('filename', filename)
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    
    execute(data, methods, x0Type, algPara, reg, mypath, prob, lamda)

def execute(data, methods, x0Type, algPara, reg, mypath, prob, lamda):  
    """
    Excute all methods/problems with 1 total run and give plots.
    """            
    
    data_dir = '../Data'  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('device', device)
    print('Dataset:', data[0])
    if data[0] == 'cifar10':
        train_Set = datasets.CIFAR10(data_dir, train=True,
                                    transform=transforms.ToTensor(), 
                                    download=True)  
        (n, r, c, rgb) = train_Set.data.shape
        d = rgb*r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = torch.DoubleTensor(X)
        Y_index = train_Set.targets  
        Y_index = torch.DoubleTensor(Y_index)
    if data[0] == 'mnist':
        train_Set = datasets.MNIST(data_dir, train=True,
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
        
        
    if data[0] == 'CelebA':
        train_Set = datasets.CelebA(data_dir, split ='train', target_type='attr',
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
        
    if data[0] == 'stl10':
        train_Set = datasets.STL10(data_dir,split='train',
                                   transform=transforms.ToTensor(), 
                                   download=True)
        (n, r, c, rgb) = train_Set.data.shape
        d = rgb*r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = torch.DoubleTensor(X)
#        print(X.shape, train_Set)
        Y_index = train_Set.labels
        Y_index = torch.DoubleTensor(Y_index)
        
    if data[0] == 'caltech256':
        train_Set = datasets.Caltech256(data_dir,
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
        
    if data[0] == 'caltech101':
        train_Set = datasets.Caltech101(data_dir,
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
        
    if data[0] == 'fmnist':
        train_Set = datasets.FashionMNIST(data_dir, train=True,
                                    transform=transforms.ToTensor(), 
                                    download=True)
        (n, r, c) = train_Set.data.shape
        d = r*c
        X = train_Set.data.reshape(n, d)
        X = X/255
        total_C = 10
        X = X.double()
        Y_index = train_Set.targets  
        Y_index = Y_index.double()
    
    if data[0] == 'gisette' or data[0] == 'arcene':
        train_X, train_Y, test_X, test_Y, idx = loaddata(data_dir, data[0])
        X = torch.from_numpy(train_X).double()
        X = X/torch.max(X)
        Y = torch.from_numpy(train_Y).double()
        Y_index = (Y+1)/2
        d = X.shape[1]
        total_C = 2
    
    if prob == 'nls' or 'logitreg' or 'student_t':
        Y = (Y_index%2!=0).double()*1
#        Y = (Y_index<5).double()*1 #88
        l = d
    if prob == 'softmax':
        I = torch.eye(total_C, total_C - 1)
#        Y = (Y_index%2==0).double()*1
        Y = I[np.array(Y_index), :]
        Y = Y.double()
        l = d*(total_C - 1)
        
    spnorm = np.linalg.norm(X, 2)
    if prob == 'softmax' or prob == 'logitreg':
        algPara.L_g=spnorm**2/6/X.shape[0] + lamda
    if prob == 'nls':
        algPara.L_g=spnorm**2/4/X.shape[0] + lamda
    if prob == 'student_t':
        algPara.L_g=2*spnorm**2/X.shape[0]/algPara.student_t_nu + lamda
    
    X = X.to(device)
    Y = Y.to(device)
    
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    print('Lipschiz', algPara.L_g)
    print('Original_Dataset_shape:', X.shape, end='  ') 
            
    if prob == 'softmax':      
        # X, Y, l = sofmax_init(train_X, train_Y)    
        obj = lambda x, control=None, HProp=1: softmax(
                X, Y, x, HProp, control, reg)  
        
    if prob == 'nls':
        # X, Y, l = nls_init(train_X, train_Y, idx=5)
        obj = lambda x, control=None, HProp=1: least_square(
                X, Y, x, HProp, control, reg)
        
    if prob == 'student_t':
        # X, Y, l = nls_init(train_X, train_Y, idx=5)
        obj = lambda x, control=None, HProp=1: student_t(
                X, Y, x, nu=algPara.student_t_nu, HProp=HProp, 
                arg=control, reg=reg)
        
    if prob == 'logitreg':
        # X, Y, l = nls_init(train_X, train_Y, idx=5)
        obj = lambda x, control=None, HProp=1: logitreg(
                X, Y, x, HProp, control, reg)
    myx0 = generate_x0(x0Type, l, zoom=algPara.zoom, dType=algPara.dType)  
    x0 = myx0.to(device)
    
    algPara.gamma = algPara.gamma/algPara.L_g/2
    if ('AAR_m_ablation' in methods) or ('AAR_m_descent' in methods):
        algPara.c2 = algPara.c2/algPara.L_g/algPara.m/2
        algPara.Andersonm = 5
        methods_all, record_all = run_algorithms(
                obj, x0, methods, algPara, mypath)
    else:
        if algPara.m > 0:
            algPara.c2 = algPara.c2/algPara.L_g/algPara.m/2
            methods_all, record_all = run_algorithms(
                    obj, x0, methods, algPara, mypath)
            showFigure(methods_all, record_all, prob, mypath)
        else:
            AAR_m = [5, 10, 15, 20, 30]
            for i in range(len(AAR_m)):
                mypath2 = mypath + '/' + ('%s'%AAR_m[i])
                if not os.path.isdir(mypath2):
                   os.makedirs(mypath2)
                algPara.Andersonm = AAR_m[i]
                algPara.c2 = algPara.c2/algPara.L_g/algPara.Andersonm/2
                methods_all, record_all = run_algorithms(
                        obj, x0, methods, algPara, mypath2)
                showFigure(methods_all, record_all, prob, mypath2)
        
def run_algorithms(obj, x0, methods, algPara, mypath):
    """
    Distribute all problems to its cooresponding optimisation methods.
    """
    record_all = []            
    record_txt = lambda filename, myrecord: np.savetxt(
            os.path.join(mypath, filename+'.txt'), myrecord.cpu(), delimiter=',') 
        
        
    if 'AAR_m_ablation' in methods:
        print(' ')
        AAR_m = [5, 10, 15, 20, 30]
        i = 0
        while i < len(AAR_m):
            record_all = []  
            AAR_mi = AAR_m[i]    
            mypath2 = mypath + '/' + ('m%s'%AAR_m[i])
            if not os.path.isdir(mypath2):
               os.makedirs(mypath2)
            if AAR_mi == 5:
                myMethod0 = 'AAR_m_descent_0%s' % (AAR_mi)
            else:    
                myMethod0 = 'AAR_m_descent_%s' % (AAR_mi)
            for j in range(3):
                myMethod = myMethod0
                gamma = algPara.gamma #
                c1 = algPara.c1 #
                c2 = algPara.c2 #
                c3 = algPara.c3 #
                if j == 1:
                    gamma = 0
                    c1 = 1E10
                    c2 = algPara.c2/0.99 #
                    c3 = 1E10
                    myMethod = myMethod0 + '_strict'
                elif j == 2:
                    gamma = algPara.gamma/0.01
                    c1 = 0
                    c2 = 0
                    c3 = 0
                    myMethod = myMethod0 + '_loose'
                print(myMethod)
                x, record = AndersonAcc(
                        obj, x0, AAR_mi, algPara.lamda, algPara.L_g, 
                        algPara.mainLoopMaxItrs, algPara.funcEvalMax, gamma, 
                        c1,  c2*algPara.Andersonm/AAR_mi, c3, algPara.nu, algPara.gradTol, algPara.show, 
                        'global', record_txt)
            
                record = record.cpu()
                if algPara.savetxt is True:
                    np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
                    
                record_all.append(myMethod)
                record_all.append(record)
            
            i += 1
            
            methods_all = record_all[::2]
            record_all = record_all[1::2]
            showFigure(methods_all, record_all, 'Exp', mypath2)
        
    if 'AAR_m_descent' in methods:
        print(' ')
        AAR_m = [5, 10, 15, 20, 30]
        i = 0
        record_all = [] 
        while i < len(AAR_m): 
            AAR_mi = AAR_m[i]               
            if AAR_mi == 5:
                myMethod = 'AAR_m_descent_0%s' % (AAR_mi)
            else:    
                myMethod = 'AAR_m_descent_%s' % (AAR_mi)
            print(myMethod)
            x, record = AndersonAcc(
                    obj, x0, AAR_mi, algPara.lamda, algPara.L_g, 
                    algPara.mainLoopMaxItrs, algPara.funcEvalMax, algPara.gamma, 
                    algPara.c1, algPara.c2*algPara.Andersonm/AAR_mi, algPara.c3, algPara.nu, algPara.gradTol, algPara.show, 
                    'global', record_txt)
            
            record = record.cpu()
            if algPara.savetxt is True:
                np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
                
            record_all.append(myMethod)
            record_all.append(record)
            
            i += 1
            
        methods_all = record_all[::2]
        record_all = record_all[1::2]
        showFigure(methods_all, record_all, 'Exp', mypath)
            
    if 'AndersonAcc' in methods:
        print(' ')
        myMethod = 'AndersonAcc'
        arg = ['pure', 'global', 'residual', 'GD']
        # arg = ['global']
        flag = 1
        i = 0
        maxOC = algPara.funcEvalMax
        while flag == 1 and i < (len(arg)):
            arg_i = arg[i]               
            myMethod = 'AA_%s (restr)' % (arg_i)
            if i == 1:
                # myMethod = 'AA_global (Alg. 3)'
                myMethod = 'AA_global (Alg. 4.1)'
            if i == 3:
                myMethod = 'GD'
            print(myMethod)
            x, record = AndersonAcc(
                    obj, x0, algPara.Andersonm, algPara.lamda, algPara.L_g, algPara.mainLoopMaxItrs, 
                    maxOC, algPara.gamma, algPara.c1, algPara.c2, algPara.c3, algPara.nu, algPara.gradTol, algPara.show, arg_i, record_txt)
            
            record = record.cpu()
                
            if algPara.savetxt is True:
                np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
                
            record_all.append(myMethod)
            record_all.append(record)
            
            if i == 0 or i == 2:    
                myMethod = 'AA_%s' % (arg_i)
                print(myMethod)
                x, record = AndersonAcc(
                        obj, x0, algPara.Andersonm, algPara.lamda, algPara.L_g, algPara.mainLoopMaxItrs, 
                        maxOC, algPara.gamma, algPara.c1, algPara.c2, algPara.c3, algPara.nu, algPara.gradTol, algPara.show, arg_i, record_txt, False)
                record = record.cpu()
                if algPara.savetxt is True:
                    np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
                
                record_all.append(myMethod)
                record_all.append(record)
            
            i += 1
    
    if 'L_BFGS' in methods:
        print(' ')
        myMethod = 'L_BFGS'
        print(myMethod)
        record_all.append(myMethod)
        x, record = L_BFGS(
                obj, x0, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                algPara.lineSearchMaxItrs, algPara.gradTol, algPara.L, algPara.beta, 
                algPara.beta2, algPara.show, record_txt)
        record = record.cpu()
        if algPara.savetxt is True:
            np.savetxt(os.path.join(mypath, 'L_BFGS.txt'), record, delimiter=',')
        record_all.append(record)
            
    methods_all = record_all[::2]
    record_all = record_all[1::2]
    
    return methods_all, record_all

        
def sofmax_init(train_X, train_Y):
    """
    Initialize data matrix for softmax problems.
    For multi classes classification.
    INPUT:
        train_X: raw training data
        train_Y: raw label data
    OUTPUT:
        train_X: DATA matrix
        Y: label matrix
        l: dimensions
    """
    n, d= train_X.shape
    Classes = sorted(set(train_Y))
    Total_C  = len(Classes)
    if Total_C == 2:
        train_Y = (train_Y == 1)*1
    l = d*(Total_C-1)
    I = np.ones(n)
    
    X_label = np.array([i for i in range(n)])
    Y = sparse.coo_matrix((I,(X_label, train_Y)), shape=(
            n, Total_C)).tocsr().toarray()
    Y = Y[:,:-1]
    return train_X, Y, l    

        
def nls_init(train_X, train_Y, idx=5):
    """
    Initialize data matrix for non-linear least square problems.
    For binary classification.
    INPUT:
        train_X: raw training data
        train_Y: raw label data
        idx: a number s.t., relabelling index >= idx classes into 1, the rest 0. 
    OUTPUT:
        train_X: DATA matrix
        Y: label matrix
        l: dimensions
    """
    n, d= train_X.shape
    Y = (train_Y >= idx)*1 #bool to int
    Y = Y.reshape(n,1)
    l = d
    return train_X, Y, l

def scale_train_X(train_X, standarlize=False, normalize=False): 
    """
    Standarlization/Normalization of trainning DATA.
    """
    if standarlize:
        train_X = preprocessing.scale(train_X)            
    if normalize:
        train_X = preprocessing.normalize(train_X, norm='l2')
    return train_X

    
def generate_x0(x0Type, l, zoom=1, dType=torch.double, dvc = 'cpu'):    
    """
    Generate different type starting point.
    """
    if x0Type == 'randn':
        x0 = torch.randn(l, dtype=dType, device=dvc)/zoom
    if x0Type == 'rand':
        x0 = torch.rand(l, dtype=dType, device=dvc)/zoom
    if x0Type == 'ones':
        x0 = torch.ones(l, dtype=dType, device=dvc)
    if x0Type == 'zeros':
        x0 = torch.zeros(l, dtype=dType, device=dvc)
    return x0