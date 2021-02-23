
import numpy as np
import pylab as plt
from matplotlib import cm
import h5py as hp
import time
import random 



Lx, Lz = 1.0, 1.0

Nx, Nz = 32, 32

hx, hz = Lx/(Nx-1), Lz/(Nz-1)

x = np.linspace(0, 1, Nx, endpoint=True)        
z = np.linspace(0, 1, Nz, endpoint=True)    

hx2, hz2 = hx*hx, hz*hz

idx2, idz2 = 1.0/hx2, 1.0/hz2

Ra = 1.0e4

Pr = 1.0

print("#", "Ra=", Ra, "Pr=", Pr)

#Ro = np.sqrt(Ra/(Ta*Pr))

nu, kappa = np.sqrt(Pr/Ra), 1.0/np.sqrt(Ra*Pr)

#print(nu, kappa)

dt = 0.02

tMax = 100

restart = 0    # 0-Fresh, 1-Restart

# Number of iterations after which output must be printed to standard I/O
opInt = 1

# File writing interval
fwInt = 2

# Tolerance value in Jacobi iterations
VpTolerance = 1.0e-5

# Tolerance value in Jacobi iterations
PoissonTolerance = 1.0e-3

gssor = 1.0

maxCount = 1e4




def getDiv(U, W):

    global Nx, Nz, hx, hz

    divMat = np.zeros([Nx, Nz])

    divMat[1:Nx-1, 1:Nz-1] = ((U[2:Nx, 1:Nz-1] - U[0:Nx-2, 1:Nz-1])*0.5/hx +
                            (W[1:Nx-1, 2:Nz] - W[1:Nx-1, 0:Nz-2])*0.5/hz)
    
    #return np.unravel_index(divMat.argmax(), divMat.shape), np.mean(divMat)
    return np.max(divMat)
    



def writeSoln(U, W, P, T, time):

    fName = "Soln_" + "{0:09.5f}.h5".format(time)
    print("#Writing solution file: ", fName)        
    f = hp.File(fName, "w")

    dset = f.create_dataset("U", data = U)
    dset = f.create_dataset("W", data = W)
    dset = f.create_dataset("T", data = T)
    dset = f.create_dataset("P", data = P)

    f.close()



def initFields():
    global Nx, Nz
    global U, W, P, T
    global Hx, Hz, Ht, Pp
    global restart

    P = np.ones([Nx ,Nz])

    #T = random.uniform(0, 1) * np.ones([L, M, N])
    T = np.zeros([Nx ,Nz])

    T[:, 0:Nz] = 1.0 - z[0:Nz]

    #print(T)

    U = np.zeros([Nx ,Nz])

    #W = random.uniform(0, 1) * np.ones([L, M, N])
    W = np.zeros([Nx ,Nz])

    time = 0

    filename = "Soln_500.00000.h5"

    if restart == 1:
        def hdf5_reader(filename,dataset):
            file_V1_read = hp.File(filename)
            dataset_V1_read = file_V1_read["/"+dataset]
            V1=dataset_V1_read[:,:,:]
            return V1
    
        U = hdf5_reader(filename, "U")
        W = hdf5_reader(filename, "W")
        P = hdf5_reader(filename, "P")
        T = hdf5_reader(filename, "T")

    writeSoln(U, W, P, T, time)

    #print(np.amax(U), np.amax(V), np.amax(W))

    # Define arrays for storing RHS of NSE
    Hx = np.zeros_like(U)
    Hz = np.zeros_like(W)
    Ht = np.zeros_like(T)   
    Pp = np.zeros_like(P)

    #if probType == 0:
        # For moving top lid, U = 1.0 on lid, and second last point lies on the wall
    #    U[:, :, -2] = 1.0
    #if probType == 1:
        # Initial condition for forced channel flow
    #    U[:, :, :] = 1.0

    #ps.initVariables()

    #if testPoisson:
    #    ps.initDirichlet()


def TimeIntegrate():

    global Nx, Nz, hx, hz, x, z, dt
    global U, W, P, T
    global Hx, Hz, Pp, Ht
    
    time = 0
    fwTime = 0.0
    iCnt = 1
    
    Hx.fill(0.0)
    Hz.fill(0.0)
    Ht.fill(0.0)

    while True:

        if iCnt % opInt == 0:

            Re = np.mean(np.sqrt(U[1:Nx-1, 1:Nz-1]**2.0 + W[1:Nx-1, 1:Nz-1]**2.0))/nu
            Nu = 1.0 + np.mean(W[1:Nx-1, 1:Nz-1]*T[1:Nx-1, 1:Nz-1])/kappa
            maxDiv = getDiv(U, W)

            print("%f    %f    %f    %f" %(time, Re, Nu, maxDiv))           


    
        Hx = computeNLinDiff_X(U, W)
        Hz = computeNLinDiff_Z(U, W)
        Ht = computeNLinDiff_T(U, W, T)  
    
    
    
    
        # Calculating guessed values of U implicitly
        Hx[1:Nx-1, 1:Nz-1] = U[1:Nx-1, 1:Nz-1] + dt*(Hx[1:Nx-1, 1:Nz-1]  - (P[2:Nx, 1:Nz-1] - P[0:Nx-2, 1:Nz-1])/(2.0*hx))
        uJacobi(Hx)
    
        # Calculating guessed values of W implicitly
        Hz[1:Nx-1, 1:Nz-1] = W[1:Nx-1, 1:Nz-1] + dt*(Hz[1:Nx-1, 1:Nz-1] - ((P[1:Nx-1, 2:Nz] - P[1:Nx-1, 0:Nz-2])/(2.0*hz)) + T[1:Nx-1, 1:Nz-1])
        wJacobi(Hz)
    
        # Calculating guessed values of T implicitly
        Ht[1:Nx-1, 1:Nz-1] = T[1:Nx-1, 1:Nz-1] + dt*Ht[1:Nx-1, 1:Nz-1]
        TJacobi(Ht)   
    
        #print(np.amax(U), np.amax(V), np.amax(W))
    
        # Calculating pressure correction term
        rhs = np.zeros([Nx, Nz])
        rhs[1:Nx-1, 1:Nz-1] = ((U[2:Nx, 1:Nz-1] - U[0:Nx-2, 1:Nz-1])/(2.0*hx) +
                               (W[1:Nx-1, 2:Nz] - W[1:Nx-1, 0:Nz-2])/(2.0*hz))/dt
    
        #ps.multigrid(Pp, rhs)
    
        Pp = PoissonSolver(rhs)
    
        # Add pressure correction.
        P = P + Pp
    
        # Update new values for U, V and W
        U[1:Nx-1, 1:Nz-1] = U[1:Nx-1, 1:Nz-1] - dt*(Pp[2:Nx, 1:Nz-1] - Pp[0:Nx-2, 1:Nz-1])/(2.0*hx)
        W[1:Nx-1, 1:Nz-1] = W[1:Nx-1, 1:Nz-1] - dt*(Pp[1:Nx-1, 2:Nz] - Pp[1:Nx-1, 0:Nz-2])/(2.0*hz)

        imposeUBCs(U)                               
        imposeWBCs(W)                               
        imposePBCs(P)                               
        imposeTBCs(T) 

        #print(np.amax(U), np.amax(W))      

        #if abs(fwTime - time) < 0.5*dt:
        if abs(time - tMax)<1e-5:
            writeSoln(U, W, P, T, time)
            Z, X = np.meshgrid(x,z)
            plt.contourf(X, Z, T, 500, cmap=cm.coolwarm)
            clb = plt.colorbar()
            plt.quiver(X, Z, U, W)
            plt.axis('scaled')
            clb.ax.set_title(r'$T$', fontsize = 20)
            plt.show()
            fwTime = fwTime + fwInt                                 


        if time > tMax:
            break   

    
        time = time + dt
    
        iCnt = iCnt + 1
    
        


    #print(U[30, 30, 30], V[30, 30, 30], W[30, 30, 30])


def computeNLinDiff_X(U, W):
    global Hx
    global Nx, Nz

    Hx[1:Nx-1, 1:Nz-1] = (((U[2:Nx, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Nz-1] + U[0:Nx-2, 1:Nz-1])/hx2 + 
                                (U[1:Nx-1, 2:Nz] - 2.0*U[1:Nx-1, 1:Nz-1] + U[1:Nx-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Nz-1]*(U[2:Nx, 1:Nz-1] - U[0:Nx-2, 1:Nz-1])/(2.0*hx) -
                              W[1:Nx-1, 1:Nz-1]*(U[1:Nx-1, 2:Nz] - U[1:Nx-1, 0:Nz-2])/(2.0*hz))

    return Hx



def computeNLinDiff_Z(U, W):
    global Hz
    global Nx, Nz

    Hz[1:Nx-1, 1:Nz-1] = (((W[2:Nx, 1:Nz-1] - 2.0*W[1:Nx-1, 1:Nz-1] + W[0:Nx-2, 1:Nz-1])/hx2 + 
                                (W[1:Nx-1, 2:Nz] - 2.0*W[1:Nx-1, 1:Nz-1] + W[1:Nx-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Nz-1]*(W[2:Nx, 1:Nz-1] - W[0:Nx-2, 1:Nz-1])/(2.0*hx) -
                              W[1:Nx-1, 1:Nz-1]*(W[1:Nx-1, 2:Nz] - W[1:Nx-1, 0:Nz-2])/(2.0*hz))


    return Hz


def computeNLinDiff_T(U, W, T):
    global Ht
    global Nx, Nz

    Ht[1:Nx-1, 1:Nz-1] = (((T[2:Nx, 1:Nz-1] - 2.0*T[1:Nx-1, 1:Nz-1] + T[0:Nx-2, 1:Nz-1])/hx2 + 
                                (T[1:Nx-1, 2:Nz] - 2.0*T[1:Nx-1, 1:Nz-1] + T[1:Nx-1, 0:Nz-2])/hz2)*0.5*kappa -
                              U[1:Nx-1, 1:Nz-1]*(T[2:Nx, 1:Nz-1] - T[0:Nx-2, 1:Nz-1])/(2.0*hx)-
                              W[1:Nx-1, 1:Nz-1]*(T[1:Nx-1, 2:Nz] - T[1:Nx-1, 0:Nz-2])/(2.0*hz))

    return Ht


#Jacobi iterative solver for U
def uJacobi(rho):
    global hx2, hz2, hz2hx2, hx2hz2, nu, dt, VpTolerance, maxCount
    global U
    global Nx, Nz

    #Up = np.zeros_like(rho)

    jCnt = 0
    while True:


        U[1:Nx-1, 1:Nz-1] =(1.0/(1+nu*dt*(idx2 + idz2))) * (rho[1:Nx-1, 1:Nz-1] + 
                                       0.5*nu*dt*idx2*(U[0:Nx-2, 1:Nz-1] + U[2:Nx, 1:Nz-1]) +
                                       0.5*nu*dt*idz2*(U[1:Nx-1, 0:Nz-2] + U[1:Nz-1, 2:Nz]))   


        imposeUBCs(U)
        
        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Nz-1] - (U[1:Nx-1, 1:Nz-1] - 0.5*nu*dt*(
                            (U[0:Nx-2, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Nz-1] + U[2:Nx, 1:Nz-1])/hx2 +
                            (U[1:Nx-1, 0:Nz-2] - 2.0*U[1:Nx-1, 1:Nz-1] + U[1:Nx-1, 2:Nz])/hz2))))
        
            #if maxErr < tolerance:
        #print(maxErr)

        if maxErr < VpTolerance:
            #print(jCnt)
            break
        
        jCnt += 1
        if jCnt > maxCount:
                print("ERROR: Jacobi not converging in U. Aborting")
                print("Maximum error: ", maxErr)
                quit()

    return U        


#Jacobi iterative solver for W
def wJacobi(rho):
    global hx2, hz2, idx2, idy2, nu, dt, VpTolerance, maxCount  
    global W
    global Nx, Nz
    
    #Wp = np.zeros_like(rho)

    #print(np.amax(rho))
    
    jCnt = 0
    while True:

        W[1:Nx-1, 1:Nz-1] =(1.0/(1+nu*dt*(idx2 + idz2))) * (rho[1:Nx-1, 1:Nz-1] + 
                                       0.5*nu*dt*idx2*(W[0:Nx-2, 1:Nz-1] + W[2:Nx, 1:Nz-1]) +
                                       0.5*nu*dt*idz2*(W[1:Nx-1, 0:Nz-2] + W[1:Nz-1, 2:Nz]))   
    
        imposeWBCs(W)


        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Nz-1] - (W[1:Nx-1, 1:Nz-1] - 0.5*nu*dt*(
                        (W[0:Nx-2, 1:Nz-1] - 2.0*W[1:Nx-1, 1:Nz-1] + W[2:Nx, 1:Nz-1])/hx2 +
                        (W[1:Nx-1, 0:Nz-2] - 2.0*W[1:Nx-1, 1:Nz-1] + W[1:Nx-1, 2:Nz])/hz2))))
    
        #if maxErr < tolerance:

        #print(maxErr)

        if maxErr < 1e-5:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return W       


#Jacobi iterative solver for T
def TJacobi(rho):
    global hx2, hz2, idx2, idy2, nu, dt, VpTolerance, maxCount  
    global T
    global Nx, Nz
    
    T = np.zeros_like(rho)
    
    jCnt = 0
    while True:

        T[1:Nx-1, 1:Nz-1] =(1.0/(1+kappa*dt*(idx2 + idz2))) * (rho[1:Nx-1, 1:Nz-1] + 
                                       0.5*kappa*dt*idx2*(T[0:Nx-2, 1:Nz-1] + T[2:Nx, 1:Nz-1]) +
                                       0.5*kappa*dt*idz2*(T[1:Nx-1, 0:Nz-2] + T[1:Nz-1, 2:Nz]))                           
    

        imposeTBCs(T)

        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Nz-1] - (T[1:Nx-1, 1:Nz-1] - 0.5*kappa*dt*(
                        (T[0:Nx-2, 1:Nz-1] - 2.0*T[1:Nx-1, 1:Nz-1] + T[2:Nx, 1:Nz-1])/hx2 +
                        (T[1:Nx-1, 0:Nz-2] - 2.0*T[1:Nx-1, 1:Nz-1] + T[1:Nx-1, 2:Nz])/hz2))))
    
        #if maxErr < tolerance:
        if maxErr < VpTolerance:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in T. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return T       



def PoissonSolver(rho):
    global hx2, hz2, idx2, idz2, PoissonTolerance, maxCount 
    global Nx, Nz
    
    
    Pp = np.zeros([Nx, Nz])
    #Pp = np.random.rand(Nx, Ny, Nz)
    #Ppp = np.zeros([L, M, N])
    
    #print(np.amax(rho))
    
    jCnt = 0
    
    while True:

        '''
        
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                for k in range(1,Nz-1):
                    Pp[i,j,k] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[i, j, k] - 
                                       idx2*(Pp[i+1, j, k] + Pp[i-1, j, k]) -
                                       idy2*(Pp[i, j+1, k] + Pp[i, j-1, k]) -
                                       idz2*(Pp[i, j, k+1] + Pp[i, j, k-1]))

        Pp[1:L-1, 1:M-1, 1:N-1] = (1.0-gssor)*Ppp[1:L-1, 1:M-1, 1:N-1] + gssor * Pp[1:L-1, 1:M-1, 1:N-1]            

        '''
           
        
        Pp[1:Nx-1, 1:Nz-1] = (1.0/(-2.0*(idx2 + idz2))) * (rho[1:Nx-1, 1:Nz-1] - 
                                       idx2*(Pp[0:Nx-2, 1:Nz-1] + Pp[2:Nx, 1:Nz-1]) -
                                       idz2*(Pp[1:Nx-1, 0:Nz-2] + Pp[1:Nz-1, 2:Nz]))   


        #Pp[1:L-1, 1:M-1, 1:N-1] = (1.0-gssor)*Ppp[1:L-1, 1:M-1, 1:N-1] + gssor*Pp[1:L-1, 1:M-1, 1:N-1]                                                                   
           
        #Ppp = Pp.copy()

        #imposePBCs(Pp)
    
        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Nz-1] -((
                        (Pp[0:Nx-2, 1:Nz-1] - 2.0*Pp[1:Nx-1, 1:Nz-1] + Pp[2:Nx, 1:Nz-1])/hx2 +
                        (Pp[1:Nx-1, 0:Nz-2] - 2.0*Pp[1:Nx-1, 1:Nz-1] + Pp[1:Nx-1, 2:Nz])/hz2))))
    
    
        #if (jCnt % 100 == 0):
            #print(maxErr)
    
        if maxErr < PoissonTolerance:
            #print(jCnt)
            #print("Poisson solver converged")
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            quit()
    
    return Pp     


def imposeUBCs(U):
    U[0, :], U[-1, :] = 0.0, 0.0
    U[:, 0], U[:, -1] = 0.0, 0.0


def imposeWBCs(W):
    W[0, :], W[-1, :] = 0.0, 0.0, 
    W[:, 0], W[:, -1] = 0.0, 0.0  

def imposeTBCs(T):
    T[0, :], T[-1, :] = T[1, :], T[-2, :]
    T[:, 0], T[:, -1] = 1.0, 0.0

def imposePBCs(P):
    P[0, :], P[-1, :] = P[1, :], P[-2, :]
    P[:, 0], P[:, -1] = P[:, 1], P[:, -2]


initFields()

TimeIntegrate()




