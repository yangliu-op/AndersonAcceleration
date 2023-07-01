"""
Optimisation algorithms,

    INPUT:
        obj: function handle of objective function, gradient and 
            Hessian-vector product
        x0: starting point
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        gradTol: stopping condition para s.t., torch.norm(gk) <= Tol
        show: print result for every iteration
    
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, step-size, direction-type]

Termination condition: norm(gradient) < gradtol. 
Otherwise either reach maximum iterations or maximum oracle calls
""" 

import torch
from time import time
from lbfgs import lbfgs
from linesearch import linesearchzoom
from Anderson import Anderson
import copy
     
global num_every_print, orcl_every_record
num_every_print = 1
orcl_every_record = 1E6
        
def myPrint(fk, gk_norm, orcl, iters, tmk, alphak=0, iterLS=0, dType=0):
    """
    A print function for every iteration.
    """
    if iters%(num_every_print*10) == 0:
        prt1 = '   iters    Time     f          ||g||         Orcl     Direction '
        prt2 = '   alphak    iterLS'
        print(prt1 + prt2)
    
    prt1 = '%8g   %8.2f' % (iters, tmk)
    prt2 = ' %8.2e     %8.2e ' % (fk, gk_norm)
    prt3 = '%8g   %8s' % (orcl, dType)
    prt4 = '%8.2f   %8g' % (alphak, iterLS)
    print(prt1, prt2, prt3, prt4)  
    
def termination(objVal, gradNorm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):  
    """
    termination condition
    """
    termination = False
    if gradNorm < gradTol or iters >= mainLoopMaxItrs or orcl >= funcEvalMax:
        termination = True
        return termination
    
def recording(matrix, v1, v2, v3, v4, v5, v6=None, v7=None, v8=None, v9=None, dType=None):    
    """
    recording matrix with row [fx, gx, oracle-calls, time, step-size, direction-type]
    and iteration as columns
    """
    v = torch.tensor([v1, v2, v3, v4, v5], device = matrix.device)
    if dType is not None: 
        if dType == 'GD':
            v = torch.cat((v, torch.ones_like(v1).reshape(1)))
        else:
            v = torch.cat((v, torch.zeros_like(v1).reshape(1)))
    if v9 is not None:
        vv = torch.tensor([v6, v7, v8, v9], device = matrix.device)
        v = torch.cat((v, vv))
    matrix = torch.cat((matrix, v.reshape(1,-1)), axis=0)  
    return matrix


def orc_call(iterSolver, HProp, iterLS=None):
    if iterLS == None:
        iterLS = 0
    return 2 + 2*iterSolver*HProp + iterLS
        
def L_BFGS(obj, x0, mainLoopMaxItrs, funcEvalMax, lineSearchMaxItrs=50, 
           gradTol=1e-10, L=10, beta=1e-4, beta2=0.4, show=True,
           record_txt=None):
    iters = 0
    orcl = 0
    x = copy.deepcopy(x0)
    fk, gk = obj(x, 'fg')
    gk_norm = gk.norm()    
    l= len(gk)   
    alphak = 1
    tmk = 0
    iterLS = 0
    
    
    # Initialize f0, g0, oracle_call0, time0, alpha0
    record = torch.tensor([fk, gk_norm, 0, 0, 1], device=x.device).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, alphak=alphak, iterLS=iterLS)
        
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break    
        
        t0 = time()  
        if iters == 0:
            p = -gk
            S = torch.empty(l, 0, device=gk.device, dtype=x0.dtype)
            Yy = torch.empty(l, 0, device=gk.device, dtype=x0.dtype)
        else:
            s = alphak_prev * p_prev
            y = gk - g_prev
            
            if S.shape[1] >= L:
                S = S[:,1:]
                Yy = Yy[:,1:]
            S = torch.cat((S, s.reshape(-1,1)), axis=1)
            Yy = torch.cat((Yy, y.reshape(-1,1)), axis=1)
            p = -lbfgs(gk, S, Yy)
#        
        #Strong wolfe's condition with zoom
        if torch.isnan(p).any():
            break
        x, alphak, iterLS, iterLS_orcl = linesearchzoom(
            obj, fk, torch.dot(gk, p), x, p, lineSearchMaxItrs, 
            c1=beta, c2=beta2, fe=funcEvalMax-orcl)
        
        g_prev = gk
        p_prev = p
        alphak_prev = alphak         

        
        fk, gk = obj(x, 'fg')
        gk_norm = gk.norm()
        iters += 1
        orcl += iterLS_orcl
        tmk += time()-t0
                
        record = recording(record, fk, gk_norm, orcl, tmk, alphak)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('L_BFGS_%s' % orcl_sh, record)
            orcl_sh = orcl_sh*10
    return x, record
    
def AndersonAcc(obj, x0, m, lamda, L, mainLoopMaxItrs, funcEvalMax, gamma,
                c1, c2, c3, nu, gradTol=1e-10, 
                show=True, arg='global', record_txt=None, restart=True):
    iters = 0
    orcl = 0
    # x = copy.deepcopy(x0)  
    x = x0.clone() 
    fk, gk = obj(x, 'fg')
    gk_norm = gk.norm()
    tmk = 0
    acc=Anderson(x,m)
    iters = 0
    flag = 0
    gk_m_norm = gk_norm
    xType = 'GD' # implies AA step got rejected
    
    D = 10**6
    epsilon = 1/D
    eta = epsilon/100
    R = 10
    nAA = 0
    RAA = 0
    t3 = 0
    lsqr_cond = 0
    acc_cond = 0
    
    g0_norm = gk_norm/L
    safeg = True
    tmk2 = 0
    
    if arg == 'global':
        # Initialize f0, g0, oracle_call0, time0, pinv_cond, diff, gkmnorm, lsqrcond, time_without_cond, dtype, 
        record = torch.tensor([fk, gk_norm, 0, 0, acc_cond, 0, 
                               gk_m_norm, lsqr_cond, 0, 0], device=x.device).reshape(1,-1)
    else:
        # Initialize f0, g0, oracle_call0, time0, alpha0, xType
        record = torch.tensor([fk, gk_norm, 0, 0, gk_m_norm, 0], device=x.device).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, dType=xType)
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break
        t0 = time()
        gx = x - gk/L # obj(gx)[1].norm()
#        gxgrad = obj(gx, 'g')
        
        if arg == 'GD':
            x = gx # original AA
#            xType = 'GD'
            xType = 'Acc' # for plot use , omit the marks of GD       
        else:
            if arg == 'pure':
                xn = acc.compute(gx)
                x = xn # pure AA
                xType = 'Acc'
                
            if arg == 'residual':
                FDRS = gx
                xDRS = FDRS
                gkDR = x - xDRS
                xn = acc.compute(gx, eta)
                if safeg or (RAA >= R):
                    if gkDR.norm() <= D*g0_norm*(nAA/R + 1)**(1+epsilon):
                        x = xn 
                        nAA = nAA + 1
                        RAA = 1
                        safeg = False
                        xType = 'Acc'
                    else:
                        x = gx
                        xType = 'GD'
                        acc.reset(x) # 
                else:
                    x = xn
                    nAA = nAA + 1
                    RAA = RAA + 1
                    xType = 'Acc'
                
            
            if arg == 'global':
                xn = acc.compute(gx, cond=False)
                lsqr_cond = acc.condnum
                acc_cond = acc.condnum # record cond(XTX)
                # xn = acc.compute(gx, cond=True)
                # acc_cond = acc.cond # record cond(XTX)
                # t3 = acc.timec
                difff = obj(x, 'diff')(xn)
                mmm = min(c1*gk_m_norm**(nu), c2*gk_m_norm**2, c3)
                if difff + gamma*gk_norm**2 < mmm: #2000
                    x = xn
                    xType = 'Acc'
                else:
                    x = gx
                    orcl += 1
                    xType = 'GD'
                    acc.reset(x)
        if restart:
            if acc.col_idx_ % (m) == m-1:
                acc.reset(x)
    #            print('reset')
                flag = 2
        
        # fkl = fk
        fk, gk = obj(x, 'fg')     
        orcl += 2
        gk_norml = gk_norm
        gk_norm = gk.norm()
        t4 = time()
        if flag == 2: # for Accgeneral
            gk_m_norm = gk_norm
            flag = 0
            
        iters += 1  
        tmk += t4-t0-t3
        if arg == 'global':
            tmk2 += t4-t0
            record = recording(record, fk, gk_norm, orcl, tmk, acc_cond, ######dtype will be here
                               obj(xn, 'diff')(gx), gk_m_norm, difff-1/2/m/L*gk_norml**2, 
                               difff+1/2/L*gk_norml**2, dType=xType)
        else:
            record = recording(record, fk, gk_norm, orcl, tmk, acc_cond, dType=xType)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('%s_%s' % ('AA', orcl_sh), record)
            orcl_sh = orcl_sh*10
    
    return x, record
