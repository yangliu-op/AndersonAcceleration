# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:47:45 2021

@author: Liu Yang
"""

import torch
from time import time
from lsqr import lsqr
import numpy as np
# from scipy.sparse.linalg import lsqr
from scipy.sparse import csc_matrix
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Anderson:
    def __init__(self, x0, num):
        self.dType = x0.dtype
        self.device = x0.device
        self.mk = num
        self.dim = len(x0)
        self.current_F_ = x0.clone()
        self.prev_dG_ = torch.zeros(self.dim, num, dtype=self.dType, device=self.device)
        self.prev_dF_ = torch.zeros(self.dim, num, dtype=self.dType, device=self.device)
        self.theta_ = torch.zeros(num, dtype=self.dType, device=self.device)
        self.dF_scale_ = torch.zeros(num, dtype=self.dType, device=self.device)
        self.dG_scale_ = torch.zeros(num, dtype=self.dType, device=self.device)
        self.M_ = torch.zeros(num, num, dtype=self.dType, device=self.device)  # num*num array
        self.current_u_ = x0.clone()
        self.iter_ = 0
        self.col_idx_ = -1
        self.timec = 0
        self.cond = 1
        self.condnum = 1
        # self.prtgamma = False

    def compute(self, g, eta=0,cond=False):
        G = g.clone()
        self.current_F_ = g - self.current_u_
#        print(self.current_u_)
        if self.iter_ == 0:
            # self.prev_dF_[0, :] = -self.current_F_
            # self.prev_dG_[0, :] = - G
            self.current_u_ = G.clone()

        else:
            self.prev_dF_[:, self.col_idx_] += self.current_F_
            self.prev_dG_[:, self.col_idx_] += G
            eps = 1e-10
            norm = self.prev_dF_[:, self.col_idx_].norm()
#            print(self.prev_dF_)
            scale = max(norm, eps)
            self.dF_scale_[self.col_idx_] = scale
            self.prev_dF_[:, self.col_idx_] /= scale
            m_k = min(self.iter_, self.mk)

            if eta == 0:
                if m_k == 1:
                    self.theta_[0] = 0
                    dF_norm = torch.linalg.norm(self.prev_dF_[:,self.col_idx_])
                    self.theta_[0] = - torch.dot(self.prev_dF_[:,self.col_idx_], self.current_F_[:])/(dF_norm**2)
                    self.cond = 1
                else:
                    result = lsqr(self.prev_dF_[:,0:m_k],  -self.current_F_)
                    # A = csc_matrix(self.prev_dF_[:,0:m_k].cpu())
                    # result = lsqr(A,  -self.current_F_.cpu())
                    # self.prtgamma = False
                    t_theta = result[0]
                    # error = (self.prev_dF_[:, 0:m_k].T@(self.prev_dF_[:,0:m_k]@ t_theta+self.current_F_)).norm()/(self.current_F_).norm()
                    # Q,R=torch.linalg.qr(self.prev_dF_[:,0:m_k])
                    # print(torch.pinverse(self.prev_dF_[:,0:m_k]))
                    # Qb = -(Q.T @ self.current_F_)
                    # t_theta1 = torch.linalg.solve(R,Qb)
                    
                    # t_theta = torch.linalg.lstsq(self.prev_dF_[:, 0:m_k], -self.current_F_)[0]
                    # error = torch.linalg.norm(self.prev_dF_[:, 0:m_k].T @ (self.prev_dF_[:, 0:m_k] @ t_theta + self.current_F_)) / torch.linalg.norm(self.current_F_)
    
                    # b = torch.mv(self.prev_dF_[:, 0:m_k].T, -self.current_F_)
                    # t_theta = -torch.pinverse(self.prev_dF_[:,0:m_k]) @ self.current_F_
                    # error = torch.linalg.norm(self.prev_dF_[:, 0:m_k].T @ (self.prev_dF_[:, 0:m_k] @ t_theta + self.current_F_)) / torch.linalg.norm(self.current_F_)
                    # if (error1<error2) & (error1<error3):
                    #      self.theta_[0:m_k]=t_theta1
                    #      self.error = error1
                    # elif error2<error3:
                    #      self.theta_[0:m_k]=t_theta2
                    #      self.error = error2
                    # else:
                    #      self.theta_[0:m_k]=t_theta3
                    #      self.error = error3
                    self.theta_[0:m_k] = t_theta
                    if cond:
                        t1 = time()
                        self.cond = torch.linalg.cond(self.prev_dF_[:,0:m_k])
                        # MM = self.prev_dF_[:,0:m_k].T @ self.prev_dF_[:,0:m_k]
                        # eigenvalue = torch.linalg.eigh(MM.T @ MM)[0]
                        # self.cond = torch.sqrt(max(abs(eigenvalue))/min(abs(eigenvalue)))
                        # self.cond = np.linalg.cond(self.prev_dF_[:,0:m_k].cpu().numpy())
                        self.timec = time() - t1
            else:
                if m_k == 1:
                    dF_norm = torch.linalg.norm(self.prev_dF_[:, self.col_idx_])
                    self.M_[0, 0] = dF_norm**2
                    # coef = self.M_[0, 0] + eta * (dF_norm ** 2 + (self.prev_dF_[self.col_idx_,:]-self.prev_dG_[self.col_idx_,:]/scale).norm()**2)
                    self.cond_num = 1
                    self.theta_[0] = -torch.dot(self.prev_dF_[:, self.col_idx_], self.current_F_[:])/dF_norm**2
                else:
                    new_inner_prod = torch.mv(self.prev_dF_[:,0:m_k].T, self.prev_dF_[:,self.col_idx_])
                    self.M_[self.col_idx_, 0:m_k] = new_inner_prod
                    self.M_[0:m_k, self.col_idx_] = new_inner_prod
                    # tt = self.prev_dF_[:,0:m_k]
                    # debug = self.M_[0:m_k,0:m_k] -tt.T @ tt
                    b = -torch.mv(self.prev_dF_[:, 0:m_k].T, self.current_F_)
                    self.theta_[0:m_k] = torch.pinverse(self.M_[0:m_k, 0:m_k] + eta * (
                            torch.norm(self.prev_dF_[:, 0:m_k],'fro')**2+torch.norm(
                                    self.prev_dF_[:, 0:m_k]-self.prev_dG_[:, 0:m_k]/self.dF_scale_[None, 
                                            0:m_k],'fro')**2 )*torch.eye(m_k, device=g.device, dtype=torch.float64)) @ b
                    # self.error = torch.linalg.norm(self.prev_dF_[:, 0:m_k].T @ (
                    #             self.prev_dF_[:, 0:m_k] @ self.theta_[0:m_k] + self.current_F_)) / torch.linalg.norm(
                        # self.current_F_)
                # self.condnum = result[5]
                # self.error = error
                # if self.error>1e-5:
                #     print('huge error=',self.error,'iter=',self.iter_)
                #     self.prtgamma = True
                # self.theta_[0:m_k] = torch.linalg.lstsq(self.prev_dF_[:,0:m_k],-self.current_F_)[0]
                #self.theta_[0:m_k] = torch.linalg.lstsq(self.M_[0:m_k, 0:m_k], b)[0]
                #self.theta_[0:m_k] = torch.pinverse(self.M_[0:m_k, 0:m_k]) @ b
            v = self.theta_[0:m_k] / self.dF_scale_[0:m_k]
#            print(self.dF_scale_[0:m_k], v)
            self.current_u_ = G + torch.mv(self.prev_dG_[:, 0:m_k], v)
        self.col_idx_ = (self.col_idx_ + 1) % self.mk
        self.prev_dF_[:, self.col_idx_] = -self.current_F_.clone()
        self.prev_dG_[:, self.col_idx_] = -G.clone()
        self.iter_ += 1

        return self.current_u_.clone()

    def replace(self, x):
        self.current_u_ = x.clone()

    def reset(self, x):
        self.current_u_ = x.clone()
        self.iter_ = 0
        self.col_idx_ = -1
        
        
def main():
    torch.manual_seed(2)
    d = 100
    W = torch.randn(d, d, dtype=torch.float64)

    A = W.T @ W
    b = torch.randn(d, dtype=torch.float64)
    f = lambda x: 0.5*torch.norm(W@x-b)**2
    g = lambda x: W.T@ (W @ x - b)
    # print(f(torch.inverse(A)@b))
    x = torch.zeros(d, dtype=torch.float64)
    L = torch.linalg.norm(A,2)
    m = 40
    maxIter = 1E5
    ng = 1
#    iters = 0
#    while iters <= maxIter:
#        gx = x - g(x)/L
#        xn = acc.compute(gx)
#        iters += 1
#        if f(xn) < f(x):
#            x = xn
#        else:
#            x = gx
#        print('f', f(x))
    iters2 = 0
    acc=Anderson(x,m)
    while iters2 <= maxIter and ng > 1E-10:
        grad = g(x)
        ng = torch.linalg.norm(grad,2)
        gx = x - grad/L
        xn = acc.compute(gx,eta=1e-11,cond=True)
        diff = 0.5*torch.sum(torch.mul(W @ (xn-gx),W @ (gx+xn)-2*b))
        if diff <= 0:
            x = xn
            print('    norm grad=', ng, iters2, acc.cond, acc.condnum)
        else:
            x = gx
            print('    norm grad=', ng)
            print('rej, cond=', acc.cond, 'iter=', acc.iter_)
            acc.reset(x)
            # print('error=',acc.error)
            print('diff=', diff)
            print('val=', f(xn))
        if iters2 % (m) == 0:
            acc.reset(x)

        iters2 = iters2 + 1
        
if __name__ == '__main__':
    main()
