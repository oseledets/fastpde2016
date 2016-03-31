import numpy as np
from DIRECT_WS_EA_RWG import DIRECT_WS_EA_RWG
from DIRECT_WS_ST_RWG import DIRECT_WS_ST_RWG


def DEMCEM_ST(k0,dx):
    ## Splitting the original rectangle in triangles for DEMCEM
    # GEOMETRY
    Np_1D = 10
    dy = dx
#    global ko
#    ko = k0
    
    r1 = np.array([0., 0., 0.])
    r2 = np.array([dx, 0., 0.])
    r3 = np.array([dx, dy, 0.])
    r4 = np.array([0., dy, 0.])
    
    
    # Evaluate ST
    I_T1_T1 = DIRECT_WS_ST_RWG(r1, r2, r3, Np_1D, k0)
    I_T2_T2 = DIRECT_WS_ST_RWG(r1, r3, r4, Np_1D, k0)
    I_T1_T2 = DIRECT_WS_EA_RWG(r1, r3, r2, r4, Np_1D, k0)
    I_T2_T1 = DIRECT_WS_EA_RWG(r1, r3, r4, r2, Np_1D, k0)
    
    I_ST = I_T1_T1 + I_T1_T2 + I_T2_T1 + I_T2_T2
    
    return I_ST
