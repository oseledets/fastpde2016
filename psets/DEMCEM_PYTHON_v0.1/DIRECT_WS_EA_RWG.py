import numpy as np
from numpy.linalg import norm
from numpy import sum, zeros, dot, cross
from math import sqrt, pi, cos, sin, tan, atan
from cmath import exp

from Gauss_1D import Gauss_1D


def DIRECT_WS_EA_RWG(r1, r2, r3, r4, Np_1D, ko):
    """
    Main body of the DIRECT EVALUATION method for the
    evaluation of the edge adjacent 4-D weakly singular integrals over planar
    triangular elements.
   
     Licensing: This code is distributed under the GNU LGPL license. 
   
     Modified:  20 September 2011
   
     Author:    Athanasios Polimeridis
     
     Ported to python: Evgeny Frolov
   
    References
   
    A. G. Polimeridis and T. V. Yioultsis, ?On the direct evaluation of weakly singular
    integrals in Galerkin mixed potential integral equation formulations,? IEEE Trans.
    Antennas Propag., vol. 56, no. 9, pp. 3011-3019, Sep. 2008.
   
    A. G. Polimeridis and J. R. Mosig, ?Complete semi-analytical treatment of weakly
    singular integrals on planar triangles via the direct evaluation method,? Int. J.
    Numerical Methods Eng., vol. 83, pp. 1625-1650, 2010.
   
    A. G. Polimeridis, J. M. Tamayo, J. M. Rius and J. R. Mosig, ?Fast and accurate
    computation of hyper-singular integrals in Galerkin surface integral equation
    formulations via the direct evaluation method,? IEEE Trans.
    Antennas Propag., vol. 59, no. 6, pp. 2329-2340, Jun. 2011.
   
    A. G. Polimeridis and J. R. Mosig, ?On the direct evaluation of surface integral
    equation impedance matrix elements involving point singularities,? IEEE Antennas
    Wireless Propag. Lett., vol. 10, pp. 599-602, 2011.
   
    INPUT DATA
    r1,r2,r3, r4 = point vectors of the triangular element's vertices
    Outer triangle P:(rp1,rp2,rp3)=(r1,r2,r3)
    Inner triangle Q:(rq1,rq2,rq3)=(r2,r1,r4)
    N_theta,N_psi = order of the Gauss-Legendre quadrature for both dimensions
    of the remaining 2-D smooth integral
   
    OUTPUT DATA
    I_DE[9]  = scalar and potential 4-D weakly singular integrals
                I_DE[0]  = I_f1_f1
                I_DE[1]  = I_f1_f2
                I_DE[2]  = I_f1_f3
                I_DE[3]  = I_f2_f1
                I_DE[4]  = I_f2_f2
                I_DE[5]  = I_f2_f3
                I_DE[6]  = I_f3_f1
                I_DE[7]  = I_f3_f2
                I_DE[8]  = I_f3_f3   
    """
    
    SQRT3 = sqrt(3)
    rp1 = r1
    rp2 = r2
    rp3 = r3
    Ap = 0.5 * norm(cross(rp2-rp1,rp3-rp1))
    Jp = Ap / SQRT3
    #
    rq1 = r2
    rq2 = r1
    rq3 = r4
    Aq = 0.5 * norm(cross(rq2-rq1,rq3-rq1))
    Jq = Aq / SQRT3
    
    ###########################################################################
    alpha = 0.5 * (r2-r1)
    beta  = 0.5 * (2*r3-r1-r2) / SQRT3
    gamma =-0.5 * (2*r4-r1-r2) / SQRT3
    # 
    coef_const, coefm_const = coefficients_const(ko)
    #
    N_theta = Np_1D
    N_psi   = Np_1D
    
    w_theta, z_theta = Gauss_1D(N_theta)
    w_psi, z_psi = Gauss_1D(N_psi)
    #
    I_const = zeros(6, dtype = complex)
    #  
    for m in range(6):
        I_theta_const = 0
        for n_theta in range(N_theta):
            theta_A, theta_B = THETA_limits(m)
            THETA = 0.5 * (theta_B-theta_A) * z_theta[n_theta] + 0.5 * (theta_B+theta_A)
            J_theta = 0.5 * (theta_B-theta_A)
            #
            tPsiA = sin(THETA) - SQRT3 * cos(THETA)
            tPsiB = sin(THETA) + SQRT3 * cos(THETA)
            #
            PsiA = atan(tPsiA)
            PsiB = atan(tPsiB)
           #
            b0 = dot(beta, beta)
            
            b1 = 2. * (dot(alpha, beta) * cos(THETA) + dot(beta, gamma) * sin(THETA))
            
            b2 = norm(alpha)**2 * cos(THETA)**2\
               + 2. * dot(alpha, gamma) * cos(THETA) * sin(THETA)\
               + norm(gamma)**2 * sin(THETA)**2
           #
            B1 = 2. * (dot(beta, gamma) * sin(THETA) - dot(alpha, beta) * cos(THETA))
            
            B2 = norm(alpha)**2 * cos(THETA)**2\
               - 2. * dot(alpha, gamma) * cos(THETA) * sin(THETA)\
               + norm(gamma)**2 *sin(THETA)**2
            #
            I_psi_const = 0.
            
            for n_psi in range(N_psi):
                psi_A, psi_B = PSI_limits(THETA,m)
                PSI = 0.5 * (psi_B-psi_A) * z_psi[n_psi] + 0.5 * (psi_B+psi_A)
                J_psi = 0.5 * (psi_B-psi_A)
                #
                B = sqrt(b0*sin(PSI)**2 + b1*cos(PSI)*sin(PSI) + b2*cos(PSI)**2)
                Bm = sqrt(b0*sin(PSI)**2 + B1*cos(PSI)*sin(PSI) + B2*cos(PSI)**2)
                #
                N,Nm = X_function_pre ( THETA, PSI, tPsiA, tPsiB, PsiA, PsiB, B, Bm, ko)
                #
                X_const = X_function_const(PSI, B, Bm, coef_const, coefm_const, N, Nm)
                #
                I_psi_const  += w_psi[n_psi] * X_const

            I_psi_const  = J_psi * I_psi_const
            I_theta_const  += w_theta[n_theta] * I_psi_const 
             

        I_const[m]  = J_theta*I_theta_const

    Iconst  = sum(I_const)
     
    # Final output
    I_DE = Jp * Jq * Iconst
    return I_DE

###########################################################################
def coefficients_const(ko):
    
    coef  = zeros(3, dtype = complex)
    coefm = zeros(3, dtype = complex)
    # 
    c = 1.
    #
    t5 = 1j ** 2
    t8 = ko ** 2
    t10 = c / t5 / t8
    #
    coef[0] = -c / 1j / ko
    coef[1] = -t10
    coef[2] = t10
    #
    cm = 1.
    #
    t1 = 1j ** 2
    t4 = ko ** 2
    t6 = cm / t1 / t4
    #
    coefm[0] = -t6
    coefm[1] = t6
    coefm[2] = -cm / 1j / ko
    ##
    
    return coef, coefm


def X_function_pre( theta, Psi, tPsiA, tPsiB, PsiA, PsiB, B, Bm, ko):
    SQRT3 = sqrt(3)
    if (Psi >= PsiB) and (Psi >= PsiA):
        D = SQRT3/sin(Psi)
        N = X2(D,B,ko)
        Nm = X2(D,Bm,ko)
    elif theta >= 0.5 * pi:
        D = SQRT3/(cos(Psi)*tPsiA)
        N = X2(D,B,ko)
        Nm = X2(D,Bm,ko)
    elif Psi >= PsiA:
        D1 = 2.*SQRT3/(cos(Psi)*(tPsiB+tan(Psi))) 
        D2 = sin(Psi)/SQRT3 
        N = X1(Psi,D1,D2,tPsiB,B,ko) 
        Nm = X1(Psi,D1,D2,tPsiB,Bm,ko) 
    else:
        D1 = SQRT3/(cos(Psi)*sin(theta))
        D2 = (cos(Psi)*tPsiA)/SQRT3
        N = X1(Psi,D1,D2,tPsiB,B,ko)
        Nm = X1(Psi,D1,D2,tPsiB,Bm,ko)

    return N, Nm
 
def X_function_const( Psi, B, Bm, coef, coefm, N, Nm):

    t2 = B ** 2
    t4 = 0.1e1 / t2 / B
    t7 = cos(Psi)
    t22 = Bm ** 2
    t24 = 0.1e1 / t22 / Bm
    # Output
    X = N[0] * t4 * coef[2] * t7 + N[1] * t4 * t7 * coef[1] + N[2] / t2 * t7 * coef[0] + Nm[0] * t24 * coefm[1] * t7 + Nm[1] * t24 * t7 * coefm[0] + Nm[2] / t22 * t7 * coefm[2]
    return X

def X1(psi,D1,D2,tpsiB,B,ko):
    SQRT3 = sqrt(3)
    N = zeros(12)

    a = 1j*ko*B
    D3 = SQRT3/(cos(psi)*tpsiB)
    aD3 = a*D3
    expaD3 = exp(-aD3)
    aD1 = a*D1
    expaD1 = exp(-aD1)
    D2a = D2/a
    H=1.-D1*D2
    T11 = (expaD3-expaD1)/aD3
    T21 = (T11-H*expaD1)/aD3
    T31 = (2*T21-H**2*expaD1)/aD3
    T41 = (3*T31-H**3*expaD1)/aD3
    T12 = D2a*(1-expaD1)
    T22 = D2a*(1-H*expaD1-T12)
    T32 = D2a*(1-H**2*expaD1-2*T22)
    T42 = D2a*(1-H**3*expaD1-3*T32)
    N21 = T11
    N31 = D3*(T11+T21)
    N81 = T21
    N91 = T31
    N101 = D3*(T21+T31)
    N111 = D3*(T31+T41)
    N121 = D3*(N101+N111)
    N41 = D3*(N31+N101)
    N51 = D3*(N41+N121)
    N22 = T12
    N32 = (T12-T22)/D2
    N82 = T22
    N92 = T32
    N102 = (T22-T32)/D2
    N112 = (T32-T42)/D2
    N122 = (N102-N112)/D2
    N42 = (N32-N102)/D2
    N52 = (N42-N122)/D2
    
    N = [1,N21+N22,N31+N32,N41+N42,N51+N52,0.5,1./3,N81+N82,N91+N92,N101+N102,N111+N112,N121+N122]
    
    return N 
 
def X2(D,B,ko):
    
    N = zeros(12)
    a = 1j*ko*B
    aD = a*D
    expaD = exp(-aD)
    T1 = (1.-expaD)/aD
    T2 = (1.-T1)/aD
    T3 = (1.-2.*T2)/aD
    T4 = (1.-3.*T3)/aD
    
    N = [1., T1, D*(T1-T2), 0., 0., 0.5, 1./3, T2, T3, D*(T2-T3), D*(T3-T4), 0.]
    N[11] = D*(N[9]-N[10])
    N[3] = D*(N[2]-N[9])
    N[4] = D*(N[3]-N[11])

    return N
 
 
def PSI_limits(theta, argument):
    SQRT3 = sqrt(3)
    PsiA=atan(sin(theta)-SQRT3*cos(theta))
    PsiB=atan(sin(theta)+SQRT3*cos(theta))
    
    if argument == 0: # I_a
        psi_A = PsiB
        psi_B = pi/2
    
    elif argument == 1: # I_b
        psi_A = PsiA
        psi_B = pi/2

    elif argument == 2: # I_c
        psi_A = 0
        psi_B = PsiA

    elif argument == 3: # I_d
        psi_A = 0
        psi_B = PsiB

    elif argument == 4: # I_d
        psi_A = PsiA
        psi_B = PsiB

    elif argument == 5: # I_d
        psi_A = 0
        psi_B = PsiA
        
    else:
        raise Exception('Wrong argument:\n{0}'.format(argument))

    return psi_A, psi_B


def THETA_limits(argument):
    
    if argument == 0: # I_a
        theta_A = 0
        theta_B = pi/2
        
    elif argument == 1: # I_b
        theta_A = pi/2
        theta_B = pi
    elif argument == 2: # I_c
        theta_A = pi/2
        theta_B = pi
        
    elif argument == 3: # I_d
        theta_A = 0
        theta_B = pi/3
        
    elif argument == 4: # I_d
        theta_A = pi/3
        theta_B = pi/2
        
    elif argument == 5: # I_d
        theta_A = pi/3
        theta_B = pi/2
    else:
        raise Exception('Wrong argument:\n{0}'.format(argument))

    return theta_A, theta_B