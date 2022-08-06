from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np
import datetime

##### MY CODE ######
class FTOCP_casadi(object):
    """ Finite Time Optimal Control Problem (FTOCP)
    Methods:
        - solve: solves the FTOCP given the initial condition x0 and terminal contraints
        - buildNonlinearProgram: builds the nonlinear program solved by the above solve methos
        - model: given x_t and u_t computes x_{t+1} = A x_t + B u_t
    """
    def __init__(self, Nb, goal, A, B, Q, R, Qf, a_max, obst, radii, safety_th, p = 0.8, printLevel = 0):
        # Define variables
        self.N = np.sum(Nb)
        self.Nb = Nb
        self.n = A.shape[1]
        self.d = B.shape[1]
        self.A = A
        self.B = B
        self.R = R
        self.Q = Q
        self.Qf = Qf
        self.a_max = a_max
        self.goal = goal

        self.th = 1 - safety_th

        self.numO = len(obst)

        self.obst = obst
        self.radii = radii

        self.P = len(Nb)
        self.numSegments = 0
        for i in range(0, self.P):
            self.numSegments += self.numO**i
            print("Total number of segments: ", self.numSegments, " N_b: ", Nb)
        
        counter = 0
        t_segment = 0
        self.Nbj = []
        self.pj = []
        for j in range(0, self.numSegments):
            if j > counter:
                t_segment += 1
                counter += self.numO**t_segment
            if t_segment<len(p):
                self.pj.append(p[t_segment])
            self.Nbj.append(Nb[t_segment])
        
        print("self.Nbj")
        print(self.Nbj)
        self.xGuessTot = np.zeros(1)

    def compute_belief(self, b0, verbose = False):
        v = [b0]
        b = [b0]

        for j in range(1, self.numSegments):
            p = self.pj[np.min([j,len(self.pj)-1])]
            self.O = []
            self.O.append(np.diag([p,1-p]))
            self.O.append(np.diag([1-p, p]))

            o = (j-1) % self.numO
            idx = (j-1) // self.numO
            v.append(self.O[o]@v[idx])
            if np.linalg.norm(v[-1],1) > 0:
                b.append(v[-1]/np.linalg.norm(v[-1],1))
            else:
                b.append(v[-1])

        self.belif = b
        self.belif_normalized = v        
        if verbose:
            print(b)
            print(v)

    def update_belief(self, bt, o):
        p = self.pj[0]
        self.O = []
        self.O.append(np.diag([p,1-p]))
        self.O.append(np.diag([1-p, p]))

        if np.sum(self.O[o]@bt) > 0:
            return self.O[o]@bt / np.sum(self.O[o]@bt)
        else:
            return self.O[o]@bt

    def solve(self, x0, verbose = False):
        # Set initail condition

        # Set box constraints on states, input and slack
        startTimer = datetime.datetime.now()

        self.lbx = [-1000]*self.tot_num_states + [-self.a_max,-self.a_max]*(self.tot_num_inputs//2) #+ [-10000]
        self.ubx = [1000]*self.tot_num_states + [ self.a_max, self.a_max]*(self.tot_num_inputs//2) #+ [10000]

        self.lbx[:self.n] = list(x0)
        self.ubx[:self.n] = list(x0)

        # Solve nonlinear programm
        if self.xGuessTot.shape[0] != len(self.lbx):
            self.xGuessTot = np.zeros(len(self.lbx))
        sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics, x0 = self.xGuessTot.tolist())
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solver_time = deltaTimer.total_seconds()
        if verbose: print("Problem solved in: ", self.solver_time, " seconds.")

        # Check solution flag
        if (self.solver.stats()['success']):
            if verbose: print("Success")
            self.feasible = 1
        else:
            print("Infeasible")
            self.feasible = 0

        # Store optimal solution
        x = np.array(sol["x"])
        self.xPredList = []
        idx_start = 0
        for j in range(self.numSegments):
            idx_end   = idx_start + (self.Nbj[j]+1)*self.n
            xOpt = x[idx_start:idx_end].reshape((self.Nbj[j]+1,self.n)).T
            self.xPredList.append(xOpt)
            idx_start = idx_end

        self.uPredList = []
        for j in range(self.numSegments):
            if self.Nbj[j] > 0:
                idx_end   = idx_start + self.Nbj[j]*self.d
                uOpt = x[idx_start:idx_end].reshape((self.Nbj[j],self.d)).T
            else:
                idx_end   = idx_start + self.d
                uOpt = x[idx_start:idx_end]
            
            self.uPredList.append(uOpt)

        self.uOpt = self.uPredList[0][:, 0]
        self.xGuessTot = x.squeeze()

    def update_Nb_and_p_sensor(self, Nb, p):
        self.P = len(Nb)
        self.numSegments = 0
        for i in range(0, self.P):
            self.numSegments += self.numO**i
            # print("Total number of segments: ", self.numSegments, " N_b: ", Nb)

        self.Nb = Nb
        counter = 0
        t_segment = 0
        self.Nbj = []
        self.pj = []
        for j in range(0, self.numSegments):
            if j > counter:
                t_segment += 1
                counter += self.numO**t_segment
            if len(p)>0 and (t_segment<len(p)):
                self.pj.append(p[t_segment])
            self.Nbj.append(Nb[t_segment])

    def update_Nb(self, Nb):
        self.P = len(Nb)
        self.numSegments = 0
        for i in range(0, self.P):
            self.numSegments += self.numO**i
            # print("Total number of segments: ", self.numSegments, " N_b: ", Nb)

        self.Nb = Nb
        counter = 0
        t_segment = 0
        self.Nbj = []
        for j in range(0, self.numSegments):
            if j > counter:
                t_segment += 1
                counter += self.numO**t_segment
            self.Nbj.append(Nb[t_segment])

    def build(self):
        # Define variables
        X_list = [SX.sym('X', (self.n, self.Nbj[j]+1)) for j in range(self.numSegments)]
        U_list = [SX.sym('X', (self.d, self.Nbj[j])) for j in range(self.numSegments)]
        # slackObs = SX.sym('X')
        # Adding equality constraints
        constraint = []
        for j in range(0, self.numSegments):
            o = (j-1) % self.numO
            if j > 0:
                idx = (j-1) // self.numO
                constraint = vertcat(constraint, X_list[j][:,0] - X_list[idx][:,-1])
    
            for i in range(0, self.Nbj[j]):
                x_next = self.A @ X_list[j][:, i] + self.B @ U_list[j][:, i]
                constraint = vertcat(constraint, X_list[j][:, i+1] - x_next)
        n_of_equality = constraint.shape[0]

        for j in range(0, self.numSegments):
            for o in range(self.numO):
                if self.belif[j][o] > self.th:
                    # print("For segment ", j, " adding constraint for ", o,". The belief is ", self.belif[j])
                    for i in range(1, self.Nbj[j]+1):
                        constraint = vertcat(constraint, ((X_list[j][self.n*i+0] - self.obst[o][0])**2/self.radii[0]**2 + (X_list[j][self.n*i+1] - self.obst[o][1])**2/self.radii[1]**2))# - slackObs)
        n_of_inequality = constraint.shape[0] - n_of_equality

        # Defining Cost
        cost = 0
        for j in range(self.numSegments):
            probability = np.sum(self.belif_normalized[j])
            for i in range(0, self.Nbj[j]):
                cost = cost + probability*( X_list[j][:, i] - self.goal ).T @ self.Q @ ( X_list[j][:, i] - self.goal )
                cost = cost + probability*( U_list[j][:, i] ).T @ self.R @ ( U_list[j][:, i] )
            
        for i in range(0, self.numO**(self.P-1)):
            idx = self.numSegments - i - 1
            probability = np.sum(self.belif_normalized[idx])                
            cost = cost + probability*( X_list[idx][:, -1] - self.goal ).T @ self.Qf @ ( X_list[idx][:, -1] - self.goal )
        # cost = cost + 1000000 * slackObs**2
        tot_var_list = []
        for x_var in X_list:
            for i in range(x_var.shape[1]):
                tot_var_list = vertcat(tot_var_list, x_var[:, i])
        self.tot_num_states = tot_var_list.shape[0]

        for u_var in U_list:
            for i in range(u_var.shape[1]):
                tot_var_list = vertcat(tot_var_list, u_var[:, i])
        self.tot_num_inputs = tot_var_list.shape[0] - self.tot_num_states
        # tot_var_list = vertcat(tot_var_list, slackObs)

        # Set IPOPT options
        opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        # opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive"}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        # opts = {"verbose":True,"ipopt.print_level":1,"print_time":1,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        nlp = {'x':tot_var_list, 'f':cost, 'g':constraint}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

        # Set lower bound of inequality constraint to zero to force: 1) n*N state dynamics and 2) obstacles
        self.lbg_dyanmics = [-0]*n_of_equality + [1]*n_of_inequality
        self.ubg_dyanmics =  [0]*n_of_equality + [10000000]*n_of_inequality
