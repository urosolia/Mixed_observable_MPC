import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib import rc
from pyparsing import alphas

class SYSTEM_ENVIRONMENT(object):
        def __init__(self, A, B, obst, radii):
            self.A = A
            self.B = B
            self.obst = obst
            self.radii = radii

        def sample_environment(self, bt):
            # Sample true environment state
            sample_uniform = np.random.uniform(0, 1)
            if sample_uniform <= bt[0]:
                self.true_environment_state = 0
            else:
                self.true_environment_state = 1

        def simulate(self, xt, ut):
            return self.A @ xt + self.B @ ut

        def get_observation(self, p_sensor):
            sample_uniform = np.random.uniform(0, 1)
            if sample_uniform <= p_sensor[0]:
                observation = self.true_environment_state
            else:
                observation = 1 - self.true_environment_state
            return observation

        def plot_predicted_trajectory(self, ftocp, time=0):
            fig = plt.figure()
            ax = plt.gca()
            o = self.obst[self.true_environment_state]
            ellipse = Ellipse(xy=(o), width=2*self.radii[0], height=2*self.radii[1], edgecolor='r', fc=(1, 0, 0, 0.2), lw=2, linewidth=2.5)
            # ellipse.set_alpha(0.5)
            ax.add_patch(ellipse)
            o = self.obst[1-self.true_environment_state]
            ellipse = Ellipse(xy=(o), width=2*self.radii[0], height=2*self.radii[1], edgecolor='g', fc='None', lw=2, linewidth=2.5)
            ax.add_patch(ellipse)

            plt.plot(ftocp.goal[0], ftocp.goal[1], 'sk', label='Goal location')

            for j in range(0, ftocp.numSegments):
                if j == 0:
                    plt.plot(ftocp.xPredList[j][0,:], ftocp.xPredList[j][1,:], '--*b', label='Optimal plan')
                    # plt.plot(ftocp.xPredList[j][0,0], ftocp.xPredList[j][1,0], 'sk', label='Initial Condition')
                else:
                    plt.plot(ftocp.xPredList[j][0,:], ftocp.xPredList[j][1,:], '--*b')
                    idx = (j-1) // ftocp.numO
                    if j == 1:
                        plt.plot([ftocp.xPredList[j][0,0], ftocp.xPredList[idx][0,-1]], [ftocp.xPredList[j][1,0], ftocp.xPredList[idx][1,-1]], '*r', label='Observation location')
                    else:
                        plt.plot([ftocp.xPredList[j][0,0], ftocp.xPredList[idx][0,-1]], [ftocp.xPredList[j][1,0], ftocp.xPredList[idx][1,-1]], '--*r')


            plt.legend(loc='lower left')
            fig.savefig('figures/predicted_traj_'+str(time)+'.pdf')
            plt.show()

        def set_true_environment_state(self, true_environment_state):
            self.true_environment_state = true_environment_state

        def plot_closed_loop(self, ftocp, xt, Nb, idx = 0):
            fig = plt.figure(figsize=(4, 3))
            ax = plt.gca()
            o = self.obst[self.true_environment_state]
            ellipse = Ellipse(xy=(o), width=2*self.radii[0], height=2*self.radii[1], edgecolor='r', fc=(1, 0, 0, 0.2), lw=2, label='True obstacle location', linewidth=2.5)
            # ellipse.set_alpha(0.5)
            ax.add_patch(ellipse)
            o = self.obst[1-self.true_environment_state]
            ellipse = Ellipse(xy=(o), width=2*self.radii[0], height=2*self.radii[1], edgecolor='g', fc='None', lw=2, linewidth=2.5)
            ax.add_patch(ellipse)


            plt.plot(xt[:, 0], xt[:, 1], '--*b', label='Closed-loop', linewidth=2.5)
            plt.plot(ftocp.goal[0], ftocp.goal[1], 'sk', label='Goal location', linewidth=2.5)
            t_counter = 0
            for t in Nb[0:-1]:
                t_counter += t
                ax.plot(xt[t_counter, 0], xt[t_counter, 1], '--*r')

            # plt.legend(loc='upper right')
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            fig.savefig('figures/closed_loop_'+str(idx)+'.pdf')
            plt.show()

        def sub_plot_closed_loop(self, ftocp, xt, ax, Nb):
            o = self.obst[self.true_environment_state]
            ellipse = Ellipse(xy=(o), width=2*self.radii[0], height=2*self.radii[1], edgecolor='r', fc=(1, 0, 0, 0.2), lw=2, label='True obstacle location')
            # ellipse.set_alpha(0.2)
            ax.add_patch(ellipse)
            o = self.obst[1-self.true_environment_state]
            ellipse = Ellipse(xy=(o), width=2*self.radii[0], height=2*self.radii[1], edgecolor='g', fc='None', lw=2)
            ax.add_patch(ellipse)

            ax.plot(xt[:, 0], xt[:, 1], '--*b', label='Closed-loop')
            ax.plot(ftocp.goal[0], ftocp.goal[1], 's', label='Goal location')

            t_counter = 0
            for t in Nb[0:-1]:
                t_counter += t
                ax.plot(xt[t_counter, 0], xt[t_counter, 1], '--*r')

            # ax.legend(loc='upper right')

        def check_collision(self, xt):
            collision = False
            for t in range(xt.shape[0]):
                if (((xt[t, 0] - self.obst[self.true_environment_state][0])**2/self.radii[0]**2) + ((xt[t, 1] - self.obst[self.true_environment_state][1])**2/self.radii[1]**2)) < 1 - 10**(-6):
                    collision = True
            return collision