import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import copy
import matplotlib.ticker as ticker
import tikzplotlib
from least_square import least_square
from regularizer import regConvex, regNonconvex
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

global colors, linestyles, markers

## for 7 figures
colors = [(0,128/255,1), (0,128/255,1), (1,128/255, 0), (1,128/255, 0), (152/255,24/255,147/255), (1,0,0), (0,0,0),(0,128/255,1), (1,0,0)]
linestyles = ['-', '--', '-', '--', '-', '-', '-',  '--', '-.']
markers = ['*', '*', '*', 'o',  'X', 'X', '*', '2','1']

def draw(plt_fun, record, label, i, NC, yaxis, xaxis=None):
    mylinewidth = 2
    if not (xaxis is not None):
        xaxis = torch.tensor(range(1,len(yaxis)+1))
    plt_fun(xaxis, yaxis, color=colors[i], 
            linestyle=linestyles[i], label = label, linewidth=mylinewidth)
    print(i)
    if NC:
        index = (record[:,5][1:] == True)
        if xaxis is not None:
            xNC = xaxis[:-1][index]
        else:
            xNC = torch.tensor(range(1,len(yaxis)))[index]
        yNC = yaxis[:-1][index]
        # plt_fun(xNC, yNC, '.', color=colors[i], marker=markers[0], markersize=8)
        plt_fun(xNC, yNC, '.', color=colors[i], marker=markers[i], markersize=12, linewidth=mylinewidth)
        
def dotdraw(plt_fun, record, label, i, NC, yaxis, xaxis=None):
    if not (xaxis is not None):
        xaxis = torch.tensor(range(1,len(yaxis)+1))
    plt_fun(xaxis, yaxis, '.', color=colors[i], 
                linestyle=linestyles[i], label = label)
    if NC:
        index = (record[:,5][1:] == True)
        if xaxis is not None:
            xNC = xaxis[:-1][index]
        else:
            xNC = torch.tensor(range(1,len(yaxis)))[index]
        yNC = yaxis[:-1][index]
        # plt_fun(xNC, yNC, '.', color=colors[i], marker=markers[0], markersize=8)
        plt_fun(xNC, yNC, '.', color=colors[i], marker=markers[i], markersize=8)
        
def showFigure(methods_all, record_all, prob, mypath, plotind=0, name=None):
    """
    Plots generator.
    Input: 
        methods_all: a list contains all methods
        record_all: a list contains all record matrix of listed methods, 
        s.t., [fx, norm(gx), oracle calls, time, stepsize, is_negative_curvature]
        prob: name of problem
        mypath: directory path for saving plots
    OUTPUT:
        Oracle calls vs. F
        Oracle calls vs. Gradient norm
        Iteration vs. Step Size
    """
    fsize = 14
    myplt2 = plt.loglog
    myplt = plt.semilogy
    
    figsz = (10,6)
    # figsz = (10,4)
    mydpi = 100
    tickssize = 25
    fig1 = plt.figure(figsize=figsz)
    
    star = True
    if star:
        F_star = record_all[0][-1,0]
        for i in range(len(methods_all)-1):
            F_star = min(F_star, record_all[i+1][-1,0])
            
    if len(record_all) == 3:
        myFname = 'Ex_F'
        myGname = 'Ex_G'
        if name[:2] =='m5':
            myxlim = [-95,3150]
            yticknum = 6
            xticknum = 5
        else:
            myxlim = [-3E1,990]
            yticknum = 6
            xticknum = 5
    else:
        star = False
        yticknum = 5
        xticknum = 4
        myxlim = [-1E2,3150]
        myFname = 'F'
        myGname = 'G'
    if name is not None:
        myFname = name + myFname
        myGname = name + myGname
    
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        record = record[:sum(record[:,2] < 3000),:]
        if star:
            record[:,0] = (record[:,0] - F_star)/max(F_star, 1)
        if methods_all[i] == 'GD' or methods_all[i] == 'AndersonAcc_pure':
            record = record[:,:5]
        draw(myplt, record, methods_all[i], plotind+i, (record.shape[1]>=6), record[:,0], record[:,2]+1)
    plt.xlabel('Oracle calls', fontsize=fsize)
    plt.ylabel(r'$\frac{f(x_k) - f^{*}}{\max \{f^{*}, 1\}}$', fontsize=fsize)
    plt.legend(fontsize=fsize)
    # plt.yscale('symlog')
    plt.ylim([5E-16,5E3])
    # plt.xlim([-1E2,900])
    plt.xlim(myxlim)
    plt.locator_params(axis='y', numticks=yticknum)
    plt.locator_params(axis='x', nbins=xticknum)
    plt.xticks(size = tickssize)
    plt.yticks(size = tickssize)
    plt.grid(True)
    fig1.savefig(os.path.join(mypath, myFname), dpi=mydpi)
    tikzplotlib.save(mypath+'/'+myFname + ".tex")
    
    
    fig2 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        record = record[:sum(record[:,2] < 3000),:]
        if methods_all[i] == 'GD' or methods_all[i] == 'AndersonAcc_pure':
            record = record[:,:5]
        draw(myplt, record, methods_all[i], plotind+i, (record.shape[1]>=6), record[:,1], record[:,2]+1)
    plt.xlabel('Oracle calls', fontsize=fsize)
    plt.ylabel(r'$|| \nabla f(x_k) ||$', fontsize=fsize)
    plt.legend(fontsize=fsize)
    # plt.yscale('symlog')
    plt.ylim([5E-8,5E1])
    plt.xlim(myxlim)
    plt.locator_params(axis='y', numticks=yticknum)
    plt.locator_params(axis='x', nbins=xticknum)
    plt.xticks(size = tickssize)
    plt.yticks(size = tickssize)
    plt.grid(True)
    fig2.savefig(os.path.join(mypath, myGname), dpi=mydpi)
    tikzplotlib.save(mypath+'/'+myGname + ".tex")


def nicefig():
    ablation = True
    # ablation = False
    # prob = 'NLS_CIFAR10'
    # prob = 'NLS_STL10'
    # prob = 'ST_CIFAR10'
    # prob = 'ST_STL10'
    for folder in os.listdir('showFig'): #only contains txt files
        print('folder', folder)
        methods_all = []
        record_all = []
        if folder[0] != 'm':
            theory()
        else:
            for method in os.listdir('showFig/'+folder):  
                if ablation:
                    if method[:5] == 'AAR_m':                
                        methods_all.append(method.rsplit('.', 1)[0])
                        print(method)
                        record = np.loadtxt(open('showFig/'+folder+'/'+method,"rb"),delimiter=",",skiprows=0)
                        record_all.append(record)
                else:
                    if method[:5] != 'AAR_m':                
                        methods_all.append(method.rsplit('.', 1)[0])
                        print(method)
                        record = np.loadtxt(open('showFig/'+folder+'/'+method,"rb"),delimiter=",",skiprows=0)
                        record_all.append(record)
            if len(methods_all) == 7:   
                plotind = 0 
                myorder = [1, 2, 3, 4, 5, 6, 0]
                methods_all = [methods_all[i] for i in myorder]
                record_all = [record_all[i] for i in myorder]
            else:
                plotind = 6
            mypath = 'showFig_plots'
            if not os.path.isdir(mypath):
               os.makedirs(mypath)
            print('lennn', len(methods_all))
            showFigure(methods_all, record_all, None, mypath, plotind, name=folder+'_')    
    

def theory():
    """
    Plots generator.
    Input: 
        methods_all: a list contains all methods
        record_all: a list contains all record matrix of listed methods, 
        s.t., [fx, norm(gx), oracle calls, time, stepsize, is_negative_curvature]
        prob: name of problem
        mypath: directory path for saving plots
    OUTPUT:
        Oracle calls vs. F
        Oracle calls vs. Gradient norm
        Iteration vs. Step Size
    """
    methods_all = []
    yticknum = 4
    xticknum = 5
    mypath = 'showFig_plots'
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    for method in os.listdir('showFig/descent'): #only contains txt files
        methods_all.append(method.rsplit('.', 1)[0])
        print(method)
        record = np.loadtxt(open('showFig/descent/'+method,"rb"),delimiter=",",skiprows=0)
        fsize = 24
        # myplt = plt.loglog
        myplt = plt.semilogy
        # myplt = plt.plot
        
        F_star = record[-1,0]
        figsz = (10,6)
        mydpi = 100
        index = (np.where(record[:,5]==1)[0]) 
        edit = 0*record[:,5] # show the previous rejection to the current node
        edit[index+int(1)] = 1
        record[:,5] = edit
        if sum(index) > 0:
            last = index[-1]
            nodes = 1
            start = last-nodes
        start = 1
        diff = record[:,6] ## fgx - fAAx
        # record = record[:sum(record[:,2] < 3000),:]
        if method[-6:-4] =='05':
            myxlim = [-95,3150]
        else:
            myxlim = [-3E1,990]
        fig2 = plt.figure(figsize=figsz)
        error = np.maximum(-diff[start:],0)/(record[:,7][start:])**3
        plt.axvline(record[:,2][index[-1]],color="red", linestyle='dashed', label=r"$10^{-19}$")
        draw(myplt, record[(start):,:], r"$f(x_k) - f(g(x_{k-1}))$", 6, (record.shape[1]>=6), error, record[start:,2]+1) ## fAAx - fgx/ norm**3
        plt.yscale('symlog')
        plt.ylim([-5E-1,1E7])
        plt.xlim(myxlim)
        plt.locator_params(axis='y', numticks=yticknum)
        plt.locator_params(axis='x', nbins=xticknum)
        plt.xticks(size = 25)
        plt.yticks(size = 25)
        plt.grid(True)
        fig2.savefig(os.path.join(mypath, 'Theory_%s'% method[-6:-4]), dpi=mydpi)
        tikzplotlib.save(mypath+'/'+'Theory_%s'% method[-6:-4] + ".tex")

        
def fixtxt():
    for method in os.listdir('showFig'): #only contains txt files\
        record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
        if method[:3] == 'AAR':
            index = np.where(record[:,1]<1E-7)
            print(index[0])
        # record2 = np.loadtxt(open('showFig/'+'zAA_global (Alg. 3).txt',"rb"),delimiter=",",skiprows=0)
        # # print('%8.10g' % record[0,4])
        # record2[:,4] = record[:,4]
        # # print('%8.10g' % record[0,4])
        # np.savetxt(os.path.join('showFig/'+'zAA_global (Alg. 3).txt'), record2, delimiter=',')
        
        
def fixH():
    # for method in os.listdir('showFig'): #only contains txt files\
    record = np.loadtxt(open('showFig/'+'AA_global (Alg. 3).txt',"rb"),delimiter=",",skiprows=0)
    record2 = np.loadtxt(open('showFig/'+'zAA_global (Alg. 3).txt',"rb"),delimiter=",",skiprows=0)
    record2[:,4] = record[:,4]
    np.savetxt(os.path.join('showFig/'+'zAA_global (Alg. 3).txt'), record2, delimiter=',')
        

def ablation():
    methods_all = []
    mypath = 'showFig_plots'
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    for method in os.listdir('showFig'): #only contains txt files
        methods_all.append(method.rsplit('.', 1)[0])
        print(method)
        record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
        fsize = 24
        # myplt = plt.loglog
        myplt = plt.semilogy
        # myplt = plt.plot
        
        figsz = (10,6)
        mydpi = 100
        index = (np.where(record[:,5]==1)[0]) 
        edit = 0*record[:,5] # show the previous rejection to the current node
        edit[index+int(1)] = 1
        record[:,5] = edit
        if sum(index) > 0:
            last = index[-1]
            nodes = 1
            start = last-nodes
        start = 1
        diff = record[start:,8] ## fAAx - fx - gnorm**2/2/m/L
        fig2 = plt.figure(figsize=figsz)
        plt.axhline(1E-20,color="red", linestyle='dashed', label=r"$10^{-19}$")
        draw(myplt, record[(start):,:], r"$f(x_AA^k) - f(x^k) - 1/2/m/L*|| nabla f(x)||^2$", 6, (record.shape[1]>=6), np.maximum(diff, 1E-20)) ## fAAx - fgx/ norm**3
        plt.grid(True)
        fig2.savefig(os.path.join(mypath, 'Ablation_rej_%s'% method[-6:-4]), dpi=mydpi)
        
        diff3 = record[start:,9] ## fAAx - fx - gnorm**2/2/m/L
        fig1 = plt.figure(figsize=figsz)
        plt.axhline(1E-20,color="red", linestyle='dashed', label=r"$10^{-19}$")
        draw(myplt, record[(start):,:], r"$-(f(x_AA^k) + f(x^k) - 1/2/L*|| nabla f(x)||^2)$", 6, (record.shape[1]>=6), np.maximum(-diff3, 1E-20)) ## fAAx - fgx/ norm**3
        plt.grid(True)
        fig1.savefig(os.path.join(mypath, 'Ablation_acc_%s'% method[-6:-4]), dpi=mydpi)
    

def cond():
    for method in os.listdir('showFig'): #only contains txt files\
        record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
        print(record[0,4])
        print('%8.4g' % record[0,4])
        print('%8.4g' % max(record[:,4]))
        np.savetxt(os.path.join('showFig/'+method), record, delimiter=',')
        
def computH():
    import torchvision.datasets as datasets
    from student_t import student_t
    import torchvision.transforms as transforms
    data_dir = '../Data'  
    train_Set = datasets.STL10(data_dir,split='train',
                               transform=transforms.ToTensor(), 
                               download=True)
    (n, r, c, rgb) = train_Set.data.shape
    d = rgb*r*c
    X = train_Set.data.reshape(n, d)
    X = X/255
    total_C = 10
    X = torch.DoubleTensor(X)
    Y_index = train_Set.labels
    Y_index = torch.DoubleTensor(Y_index)
    Y = (Y_index%2!=0).double()*1
    lamda = 1E-1
    l = d
    reg = None
    # obj = lambda x, control=None, HProp=1: least_square(
    #         X, Y, x, HProp, control, reg)
    obj = lambda x, control=None, HProp=1: student_t(
            X, Y, x, nu=20, HProp=HProp, 
            arg=control, reg=reg)
    for method in os.listdir('showFig'): #only contains txt files
        if method == 'xT.txt':
            xx = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
            print('1')
            eigmin = obj(torch.tensor(xx, dtype=(torch.float64)), 'H') + lamda
            print(eigmin)
            
        if method == 'AA_global (Alg. 3).txt':
            record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
            record[0,4] = eigmin
            np.savetxt(os.path.join('showFig/'+method), record, delimiter=',')
            
# =============================================================================
    """
    regenerate any 1 total run plot via txt record matrix in showFig folder.
    Note that directory only contains txt files.
    For performace profile plots, see pProfile.py
    """
    # showFigure(methods_all, record_all, None, mypath)
        
        
        
if __name__ == '__main__':
    # fixH()
    # fixtxt()
    nicefig()
    # ablation()
    # computH()
    # cond()
    # theory()
    # ablation_extreme()