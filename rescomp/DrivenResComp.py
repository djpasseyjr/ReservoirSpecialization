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
        true_states      = out_signal(t).T
        if self.solver == "least squares":
            self.W_out = np.linalg.lstsq(driven_states, true_states, rcond=None)[0].T

        else:
            ridge_regression = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge_regression.fit(driven_states,true_states.T)
            self.W_out       = ridge_regression.coef_

        error = np.mean(np.linalg.norm(self.W_out @ driven_states.T - true_states, ord=2,axis=0))
        self.is_trained = True

        if return_states:
            # Return node states
            return error, driven_states

        return error

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
