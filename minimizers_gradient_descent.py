from constant import gamma, damping_constant, pi, atan2, eps
import numpy as np
from field_calculation import FieldCalcutaton
import os

curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")


class Minimizers:
    """
    solvers for gradient descent
    """

    def llg_equation_rhs(self, magnetization, calculate_effective_field, zeeman_field):
        """
        compute right side of the Landau Lifshitz Gilbert equation
        https://en.wikipedia.org/wiki/Landau%E2%80%93Lifshitz%E2%80%93Gilbert_equation

        @param zeeman_field:
        @param calculate_effective_field:
        @type magnetization: object
        """
        total_field = calculate_effective_field(magnetization) + zeeman_field
        llg_right_hand_side = - gamma / (1 + damping_constant ** 2) * np.cross(magnetization, total_field) - \
                              damping_constant * gamma / (1 + damping_constant ** 2) * np.cross(magnetization, np.cross(magnetization, total_field))

        return llg_right_hand_side


    # compute llg step using Euler method
    def llg_euler_evolver(self, magnetization, dt, zeeman_field=0.0):
        """
        In mathematics and computational science, the Euler method (also called forward Euler method)
        is a first order numerical procedure for solving ordinary differential equations (ODEs) with a given initial value.
        It is the most basic explicit method for numerical integration of ordinary differential equations and is the simplest Runge Kutta method.
        The Euler method is named after Leonhard Euler, who treated it in his book Institutionum calculi integralis (published 1768 1870)
        https://en.wikipedia.org/wiki/Euler_method

        @type magnetization: object
        """
        field_cal = FieldCalcutaton()
        dm_dt = self.llg_equation_rhs(magnetization, field_cal.calculate_effective_field, zeeman_field)
        # print ("error = %g" % (dm_dt * dt).mean())
        magnetization += dt * dm_dt

        return magnetization / np.repeat(np.sqrt((magnetization * magnetization).sum(axis = 3)), 3).reshape(magnetization.shape)


    # compute llg step using RK4
    def llg_rk4_evolver(self, magnetization, dt, zeeman_field=0.0):
        """
        In numerical analysis, the Rung Kutta methods are a family of implicit and explicit iterative methods,
        which include the well-known routine called the Euler Method,
        used in temporal discretization for the approximate solutions of ordinary differential equations.
        These methods were developed around 1900 by the German mathematicians Carl Runge and Wilhelm Kutta.
        The most widely known member of the Runge Kutta family is generally referred to as "RK4",
        the "classic Runge Kutta method" or simply as "the Runge Kutta method".
        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

        @type magnetization: object
        """
        field_cal = FieldCalcutaton()
        k1 = self.llg_equation_rhs(magnetization, field_cal.calculate_effective_field, zeeman_field)
        k2 = self.llg_equation_rhs(magnetization + (dt / 2.0) * k1, field_cal.calculate_effective_field, zeeman_field)
        k3 = self.llg_equation_rhs(magnetization + (dt / 2.0) * k2, field_cal.calculate_effective_field, zeeman_field)
        k4 = self.llg_equation_rhs(magnetization + dt * k3, field_cal.calculate_effective_field, zeeman_field)

        dm_dt = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        magnetization += dt * dm_dt
        magnetization = magnetization / np.repeat(np.sqrt((magnetization * magnetization).sum(axis = 3)), 3).reshape(magnetization.shape)

        return magnetization


    def cost_func(self, magnetization_curr, magnetization_prev):
        """
        The cost_func is defined as the max of (dm_curr - dm_prev for each spin in the 3D system)

        @type magnetization_curr: current prediction
        @type magnetization_prev: last prediction
        """
        n_x = len(magnetization_curr)
        n_y = len(magnetization_curr[0])
        n_z = len(magnetization_curr[0][0])
        max_diff = float("-inf")
        for x in range(n_x):
            for y in range(n_y):
                for z in range(n_z):
                    diff_sq = 0
                    for i in range(3):
                        diff_sq += (magnetization_curr[x][y][z][i] - magnetization_prev[x][y][z][i]) ** 2
                    diff = np.sqrt(diff_sq)
                    max_diff = max(diff, max_diff)

        return max_diff


    def gradient_descent(self, magnetization, dt, zeeman_field=0.0):
        """
        gradient descent method to compute the equilibrium state of the 3D system

        @type magnetization: object
        """
        magnetization_prev = np.copy(magnetization)
        self.llg_rk4_evolver(magnetization, dt, zeeman_field)

        cost = self.cost_func(magnetization, magnetization_prev)

        costs = []
        costs.append(cost)

        while cost > eps:
            # print(cost)
            magnetization_prev = np.copy(magnetization)
            self.llg_rk4_evolver(magnetization, dt, zeeman_field)
            cost = self.cost_func(magnetization, magnetization_prev)
            costs.append(cost)

        np.savetxt(curr_folder + "\costs_vs_time.dat", costs)
        # print "cost = %e" % (cost)
