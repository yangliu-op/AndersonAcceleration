from initialize import initialize
import torch

class algPara():
    def __init__(self, value):
        self.value = value

def main(data, prob, lamda, m):
    #initialize methods
    
    methods = [
        
        
        #### Comparisons
            'AndersonAcc',
            'L_BFGS',
            
            
        #### Theorems
            # 'AAR_m_descent', # for m = 5, 10, 15, 20, 30
            # 'AAR_m_ablation', # extreme ablation m = 5, 10, 15, 20, 30
            
            
            ]
    
    algPara.regType = [
    #        'None',
            'Convex',
    #        'Nonconvex',
            ] 
    
    #initial point
    x0Type = [
            'randn',
            # 'rand',
            # 'ones',
            # 'zeros',
            ]
    #initialize parameter
    algPara.funcEvalMax = 3E3 #Set mainloop stops with Maximum Function Evaluations
    algPara.mainLoopMaxItrs = 1E5 #Set mainloop stops with Maximum Iterations
    algPara.gradTol = 1e-7 #If norm(g)<gradTol, minFunc loop breaks
    algPara.lineSearchMaxItrs = 1E3
    algPara.L = m
    algPara.m = m ## m=0 execute m = 5, 10, 15, 20, 30
    algPara.beta = 1E-4
    algPara.beta2 = 0.9
    algPara.lamda = lamda
    # lamda = 1E-2 #regularizer
    algPara.zoom = 1
    algPara.student_t_nu = 20
    # algPara.cutData = 100
    algPara.dType = torch.float64
    algPara.savetxt = True
    algPara.show = False
    algPara.Andersonm = algPara.L
    algPara.gamma = 0.01 #### /Lg/2 later
    algPara.c2 = 0.99 #### /Lg/m/2 later
    algPara.c1 = 1 #
    algPara.c3 = 1 #
    algPara.nu = 2.1    
    
    ## Initialize
    if hasattr(algPara, 'seed'):
        initialize(data, methods, prob, x0Type, algPara, lamda, algPara.seed)
    else:
        initialize(data, methods, prob, x0Type, algPara, lamda)
        
    
if __name__ == '__main__':
    m = 10 ## m=0 execute m = 5, 10, 15, 20, 30
    
    ## Dataset, problem, regularization lambda, m
    # main(['cifar10'], ['student_t'], 1E-2, m)  
    main(['cifar10'], ['nls'], 1E-2, m)
    # main(['stl10'], ['student_t'], 1E-1, m)  
    # main(['stl10'], ['nls'], 1E-1, m)
    
    
    
    