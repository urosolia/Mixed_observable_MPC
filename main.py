import numpy as np
from src.ftocp_casadi import FTOCP_casadi
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib import rc
from src.system_environment import SYSTEM_ENVIRONMENT
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('font', size=12)
import os
if not os.path.exists('figures'):
	os.makedirs('figures')

# ==============================================================================================
# Initial Conditions
x0     = np.array([-4,0,0,0])
bt     = [np.array([0.5, 0.5])]
# bt     = [np.array([0.2, 0.8])]

# ==============================================================================================
# Define system dynamics
dt = 0.1
A = np.array([[1, 0, dt,  0], 
	          [0, 1,  0, dt],
	          [0, 0,  1,  0],
	          [0, 0,  0,  1]])

B = np.array([[ 0, 0], 
	          [ 0, 0],
	          [dt, 0],
	          [ 0, dt]])

n = A.shape[1]; d = B.shape[1]

# ==============================================================================================
# Define MPC parameters
Q  = 0.1  * np.eye(n)
Q[1,1] = 10.0

R  = 1.0 * np.eye(d)
Qf = 1000.0 * np.eye(n)
Nb = [4, 4, 18]

# ==============================================================================================
# Define environment parameters
obst = []
obst.append([7, -0.2])
obst.append([6,  0.2])

width = 5
highth = 1.5
radii = [width/2, highth/2]

a_max = 20.0
goal = np.array([14.0,  0.0, 0.0, 0.0])

# Initialize environemnt and MPC problem
p_sensor = [0.7, 0.9]
safety_th = 0.85
system_environment = SYSTEM_ENVIRONMENT(A, B, obst, radii)

ftocp = FTOCP_casadi(Nb, goal, A, B, Q, R, Qf, a_max, obst, radii, safety_th, p=p_sensor, printLevel = 0)


# ==============================================================================================
# Start solving control task
n_of_trials = 10
plot_flag = False
collision_counter = 0
closed_loop = []
true_state_list = []
measurement_list = []
solver_time = []
# Trial loop
for trial in range(n_of_trials):
	# Initialize belief, state, and observation list
	converged = False
	bt = [bt[0]]
	xt = [x0]
	observation_list = []


	# Initialize horizon and update MPC belief
	Nb_t = Nb[:]
	p_sensor_t = p_sensor[:]
	ftocp.update_Nb_and_p_sensor(Nb_t, p_sensor_t)
	ftocp.compute_belief(bt[-1], verbose=True)

	# Randomly sample environment configuration
	system_environment.sample_environment(bt[-1])
	print("============ True state ", system_environment.true_environment_state, " for trial ", trial)

	# Time loop
	while (converged == False):
		# Solve MPC problem
		ftocp.update_Nb(Nb_t)
		ftocp.build()
		ftocp.solve(xt[-1])
		solver_time.append(ftocp.solver_time)

		# Plot
		if plot_flag == True:
			system_environment.plot_predicted_trajectory(ftocp, len(xt))

		# Simulate
		uOpt = ftocp.uOpt
		x_next = system_environment.simulate(xt[-1], uOpt)
		xt.append(x_next)

		# Update the horizon
		if Nb_t[0] == 1:
			# Sample observation
			observation = system_environment.get_observation(ftocp.pj)
			observation_list.append(observation)
			
			# Update belief
			bt_next = ftocp.update_belief(bt[-1], observation)
			bt.append(bt_next)

			# Reduce horizon and update belief
			Nb_t.pop(0)
			ftocp.compute_belief(bt[-1])
			p_sensor_t.pop(0)

			ftocp.update_Nb_and_p_sensor(Nb_t, p_sensor_t)
		else:
			# Update horizon
			Nb_t[0] = Nb_t[0]-1
			Nb_t[-1] = Nb_t[-1]+1

		# Check convergence
		if np.linalg.norm(x_next - ftocp.goal, 2) <= 0.5:
			xt = np.array(xt)
			converged = True
			closed_loop.append(xt)
			true_state_list.append(system_environment.true_environment_state)
			measurement_list.append(observation_list)
			if plot_flag == True:
				system_environment.plot_closed_loop(ftocp, xt, Nb)
			if system_environment.check_collision(xt) == True:
				print("Collision!")
				collision_counter += 1
			print("Reached Convergence. Obervation list: ", observation_list, ". True state ", system_environment.true_environment_state)

print("Total number of collision ", collision_counter, " out of ", n_of_trials, "trials")
print("Avg solver time: ", np.mean(solver_time), ". Max solver time: ", np.max(solver_time), ". Min solver time: ", np.min(solver_time))

# Plot closed-loop trajectories
for i in range(2):
	for j in range(4):
		for xt, true_state, obs in zip(closed_loop, true_state_list, measurement_list):
			if (true_state == i) and int(''.join(str(i) for i in obs), 2)==j:
				system_environment.set_true_environment_state(true_state)
				# system_environment.sub_plot_closed_loop(ftocp, xt, axs[i][j], Nb)
				system_environment.plot_closed_loop(ftocp, xt, Nb, idx=str(obs)+'_'+str(true_state))
				break

plt.show()