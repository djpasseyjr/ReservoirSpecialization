from rescomp.ResComp import *

class DrivenResComp(ResComp):
    """ Reservoir Computer that learns a response to a input signal
    """
    def __init__(self, *args, drive_dim=1, delta=.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.drive_dim = drive_dim
        self.delta = delta
        self.W_drive = np.random.rand(self.res_sz, self.drive_dim) - 1


    def drive(self, t, out_signal, drive_signal):
        """ Parameters
            t (1 dim ndarray):
                an array of time values
            out_signal (function):
                for each i, out_signal(t[i]) produces the desired output signal
                at timesetp i
            drive_signal  (function):
                for each i, drive_signal(t[i]) produces the driving input signal
                at timesetp i
        """
        # Reservoir ode
        def res_f(r,t):
            transform_out = self.sigma*self.W_in @ out_signal(t)
            transform_in = self.delta*self.W_drive @ drive_signal(t)
            return self.gamma*(-1*r + self.activ_f(self.res @ r + transform_out + transform_in))

        r_0    = self.state_0
        states = integrate.odeint(res_f,r_0,t)
        self.state_0 = states[-1]
        return states


    def fit(self, t, out_signal, drive_signal, return_states=False):
        """ Parameters
        t (1 dim ndarray):
            an array of time values
        out_signal (function):
            for each i, out_signal(t[i]) produces the desired output signal
            at timesetp i
        drive_signal  (function):
            for each i, drive_signal(t[i]) produces the driving input signal
            at timesetp i
        Returns
        err (float) : Error in the fit (norm of residual)
            Optionally returns: drive_states : (ndarray) of node states
        """
        driven_states    = self.drive(t, out_signal, drive_signal)
        true_states      = out_signal(t)
        self.W_out = self.solve_wout(driven_states, true_states)
        self.is_trained = True
        # Compute error
        diff = self.W_out @ driven_states.T - true_states.T
        error = np.mean(np.linalg.norm(diff, ord=2,axis=0))

        if return_states:
            # Return node states
            return error, driven_states

        return error

    def fit_batch(self, t, u, u_drive, time_window, overlap=0.0, return_states=False):
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
        """
        internal_states, target = self.drive_batch(t, u, u_drive, time_window, overlap=overlap)
        # Solve for W_out
        self.W_out = self.solve_wout(internal_states, target)
        self.is_trained = True
        # Compute error
        diff = self.W_out @ internal_states.T - target.T
        error = np.mean(np.linalg.norm(diff, ord=2, axis=0))
        if return_states:
            # Return node states
            return error, internal_states
        return error

    def drive_batch(self, t, u, u_drive, time_window, overlap=0.0):
        """ Convert signal to a batch of signals and drive the reservoir with each one.
            Concatenate the output signals and return as one array each of states and targets.
        """
        internals = ()
        targets = ()
        # Dispatch on types of u and t
        if isinstance(u, list) and isinstance(t, list) and isinstance(u_drive, list):
            for time, signal, dr_signal in zip(t, u, u_drive):
                internal, target = self._drive_batch(time, signal, dr_signal, time_window, overlap=overlap)
                internals += internal
                targets += target
        else:
            internals, targets = self._drive_batch(t, u, u_drive, time_window, overlap=overlap)
        # Stack internal states and targets
        internals = np.vstack(internals)
        targets = np.vstack(targets)
        return internals, targets

    def _drive_batch(self, t, u, u_drive, time_window, overlap=0.0):
        ts = self._partition(t, time_window, overlap=overlap)
        internals = ()
        targets = ()
        for time in ts:
            assert (time[-1] - time[0]) < time_window, f"Subarray covers {time[-1] - time[0]} seconds but `time_window` is {time_window}"
            # Set initial condition
            self.state_0 = self.W_in @ u(time[0])
            internals += (self.drive(time, u, u_drive),)
            targets += (u(time),)
        return internals, targets

    def predict(self, t, drive_signal, u_0=None, r_0=None, return_states=False):
        """
        Parameters
        t (1 dim ndarray):
            an array of time values
        drive_signal  (function):
            for each i, drive_signal(t[i]) produces the driving input signal
            at timesetp i
        Returns
        pred (ndarray) : predicted output signal values at the given times
            Optionally returns: internal_states : (ndarray) of node states
        """
        if not self.is_trained:
            raise Exception("Reservoir is untrained")

        # Reservoir prediction ode
        def res_pred_f(r,t):
            transform_in = self.delta*self.W_drive @ drive_signal(t)
            return self.gamma*(-1*r + self.activ_f(self.res @ r + self.sigma * self.W_in @ (self.W_out @ r) + transform_in))

        if r_0 is None and u_0 is None:
            r_0  = self.state_0

        elif r_0 is None and u_0 is not None:
            r_0 = self.W_in @ u_0

        internal_states = integrate.odeint(res_pred_f, r_0, t)
        pred = self.W_out @ internal_states.T

        if return_states:
            return pred, internal_states.T
        return pred
