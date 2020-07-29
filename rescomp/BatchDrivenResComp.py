from rescomp.DrivenResComp import *
from scipy.linalg import lu_factor, lu_solve

class BatchDrivenResComp(DrivenResComp):
    """ Reservoir Computer that breaks training into batches to save memory and
        allow for online updates. Training algorithm is outlined in appendix A
        of:
        Backpropagation Algorithms and Reservoir Computing in Recurrent Neural
        Networks for the Forecasting of Complex Spatiotemporal Dynamics
            Vlachas, Pathak, Hunt, Sapsis, Girvan, Ott and Koumoutsakos
    """
    def __init__(self, *args, batchsize=2000, **kwargs):
        super().__init__(*args, **kwargs)
        # Store matrixes used to compute W_out
        self.Hbar = np.zeros((self.res_sz, self.res_sz))
        self.Ybar = np.zeros((self.signal_dim, self.res_sz))
        self.batchsize = batchsize
        self.W_out_factors = None

    def drive(self, time, u, u_drive):
        """ Drive the reservoir with the u and collect state information into
            self.Hbar and self.Ybar
            Parameters
            t (1 dim array): array of time values
            u (function): for each i, u(t[i]) produces the state of the system
                that is being learned at time t[i]
            u_drive (function): for each i, u_drive(t[i]) produces the driving
                signal at time t[i]
        """
        # The i + batchsize + 1 final index adds one timestep of overlap to provide
        # the initial condition for the next batch. Overlap is removed after
        # the internal states are generated
        ts = [time[i : i + self.batchsize + 1] for i in range(0, len(time), self.batchsize)]
        # Set initial condition for reservoir nodes
        self.state_0 = self.W_in @ u(time[0])
        for t in ts:
            states = super().drive(t, u, u_drive)
            # Get next initial condition and trim overlap
            states, self.state_0 = states[:-1, :], states[-1, :]
            # Update Hbar and Ybar
            self.Hbar += states.T @ states
            Y = u(t[:-1])
            Y = self.time_dim_0(t[:-1], u, Y) # Ensure number of columns == signal dimension
            self.Ybar += Y.T @ states # Trim overlap
        # Old W_out factors if any are now incorrect. They must be recomputed
        self.W_out_factors = None

    def solve_wout(self):
        """ Solve the Tikhonov regularized least squares problem (Ridge regression)
            for W_out (The readout mapping)
        """
        W_out = self.Ybar @ np.linalg.inv(self.Hbar + self.ridge_alpha * np.eye(self.res_sz))
        return W_out

    def factor_wout(self):
        """ LU decompostion of the inverse factor of the Tikhonov least squares (see `solve_wout`)
            to avoid inverse computation
        """
        self.W_out_factors = lu_factor(self.Hbar + self.ridge_alpha * np.eye(self.res_sz))

    def fit(self, t, u, u_drive, inv_free=False):
        """
        Parameters
        t (1 dim ndarray)    : an array of time values
        u (function): for each i, u(t[i]) produces the state of
            the system that is being learned at time t[i]
        u_drive (function): for each i, u_drive(t[i]) produces the driving
            signal at time t[i]
        inv_free (Bool): Defaults to false. If true, the function will not compute W_out.
            Calling `self.predict` with inv_free=True will compute `W_out @ x` using LU
            decomposition and gaussian elimination to avoid inverse computation in
            the Tikonov least squares
        """
        self.drive(t, u, u_drive)
        if not inv_free:
            self.W_out = self.solve_wout()
            self.is_trained = True

    def fit_batch(self, t, u, u_drive, time_window, overlap=0.0, inv_free=False):
        """ Train reservoir computer on the signal as a batch of short signals instead
            of on one long continuous signal

            Parameters
            t (1 dimensional array or list of one dimensional arrays): Time values corresponding to signal u
            u (function or list of functions that accept time values): Input signal
            u_drive (function or list of functions that accept time values): Driving signal
            time_window (float): How long to make each shorter signal in time units. (If t is in seconds
                setting time_window to 2.0 will mean that each shorter signal is 2 seconds long
            overlap (float): Less than one, greater or equal to zero. (Defaults to zero) Percent that
                each signal window overlaps the previous signal window
            inv_free (Bool): Defaults to false. If true, the function will not compute W_out.
                Calling `self.predict` with inv_free=True will compute `W_out @ x` using LU
                decomposition and gaussian elimination to avoid inverse computation in
                the Tikonov least squares
        """
        if isinstance(t, list) and isinstance(u, list) and isinstance(u_drive, list):
            for time, signal, drive_signal in zip(t, u, u_drive):
                ts = self._partition(time, time_window, overlap=overlap)
                for t in ts:
                    self.drive(t, signal, drive_signal)
        else:
            ts = self._partition(t, time_window, overlap=overlap)
            for t in ts:
                self.drive(t, u, u_drive)
        if not inv_free:
            # Solve for W_out
            self.W_out = self.solve_wout()
            self.is_trained = True

    def res_pred_f(self, r,t):
        """ Reservoir prediction ode. Assumes precomputed W_out """

        return self.gamma*(-1*r + self.activ_f(self.res @ r + self.sigma * self.W_in @ (self.W_out @ r)))

    def inv_free_pred_f(self, r, t):
        """ Avoid inverse computation by solving the linear system with precomputed LU decomp.
            The const is 2n^2 for each function evaluation compared to an initial n^3
            inverse computation and an n^2 cost afterward for each function evaluation.

            This method is to be used when there are frequent updates to the Ybar and Hbar
            matrixes and repeated inverse computation is undesireable.
        """
        u_hat = self.Ybar @ lu_solve(self.W_out_factors, r)
        return self.gamma*(-1*r + self.activ_f(self.res @ r + self.sigma * self.W_in @ u_hat))

    def predict(self, t, u_drive, u_0=None, r_0=None, return_states=False, inv_free=False):

        # Define reservoir odes using u_drive
        def res_pred_f(r,t):
            transform_in = self.delta*self.W_drive @ u_drive(t)
            return self.gamma*(-1*r + self.activ_f(self.res @ r + self.sigma * self.W_in @ (self.W_out @ r) + transform_in))

        def inv_free_pred_f(r, t):
            """ Reservoir ode that avoids inverse computation """
            transform_in = self.delta*self.W_drive @ u_drive(t)
            u_hat = self.Ybar @ lu_solve(self.W_out_factors, r)
            return self.gamma*(-1*r + self.activ_f(self.res @ r + self.sigma * self.W_in @ u_hat + transform_in))

        # Determine initial condition
        if r_0 is None and u_0 is None:
            r_0  = self.state_0
        elif r_0 is None and u_0 is not None:
            r_0 = self.W_in @ u_0
        # Use precomputed inverse or not
        if inv_free:
            if self.W_out_factors is None:
                self.factor_wout()
            states = integrate.odeint(inv_free_pred_f, r_0, t)
            pred = self.Ybar @ lu_solve(self.W_out_factors, states.T)

            if return_states:
                return pred, states.T
            return pred
        else:
            if not self.is_trained:
                raise Exception("Reservoir is untrained")
            states = integrate.odeint(res_pred_f, r_0, t)
            pred = self.W_out @ states.T
        # Return internal states as well as predicition or not
            if return_states:
                return pred, states.T
            return pred
