import numpy as np
from sklearn.linear_model import Ridge
from scipy import integrate
from scipy import sparse
from specialize import *

class ResComp:
    """ Reservoir Computer Class
        Internal functionality based on:
            Attractor reconstruction by machine learning
            https://aip.scitation.org/doi/10.1063/1.5039508

        Initialization options:

        0 arguments: Initializes the reservoir as a random graph with all other
                     datamembers determined by keyword arguments
        1 argument:  Assumes argument to be an adjacency matrix. Makes the internal
                     reservoir equal to the argement.

    """
    def __init__(self, *args,
                 res_sz=200,   activ_f=np.tanh,
                 connect_p=.01, ridge_alpha=.00001,
                 spect_rad=.9, sparse_res=False,
                 sigma=0.1,    uniform_weights=False,
                 gamma=1.,     solver="ridge regression",
                 signal_dim=1
                ):

        num_in = signal_dim
        num_out = signal_dim
        res_args = (res_sz, connect_p)

        if len(args)==1:
            A = args[0]
            res_sz = A.shape[0]
            res_args = (A,)

        if len(args) == 2:
            num_in, num_out = args
        # end

        # Initialize with a supplied reservoir adj matrix
        if len(args) == 3:
            A, num_in, num_out = args
            res_args = (A,)
        # end

        # Set model attributes
        self.W_in        = np.random.rand(res_sz, num_in) - .5
        self.W_out       = np.zeros((num_out, res_sz))
        self.gamma       = gamma
        self.sigma       = sigma
        self.activ_f     = activ_f
        self.ridge_alpha = ridge_alpha
        self.state_0     = np.random.rand(res_sz)
        self.solver      = solver
        self.sparse_res  = sparse_res
        self.spect_rad   = spect_rad
        self.is_trained  = False
        self.uniform_weights = uniform_weights


        if self.sparse_res:
            self.sparse_init(*res_args)
        else:
            self.dense_init(*res_args)
        # end

    # end

    def sparse_init(self, *args, scale_rad=True):
        """ Initialize a sparse reservoir with the adjacency matrix of and Erdos-Renyi
            random graph. Remove self edges. Use uniform weights or integer weights.
            Scale the spectral radius.
        """
        # Create random reservoir
        if len(args) == 2:
            res_sz, connect_p, = args
            # Make reservoir
            weights = lambda x: np.ones(x) if self.uniform_weights else np.random.rand(x)*2
            self.res = sparse.random(res_sz,res_sz,
                                             density=connect_p,
                                             dtype=float,
                                             format="lil",
                                             data_rvs=weights)
        # end
        # Create reservoir from supplied adj matrix
        if len(args) == 1:
            A = args[0]
            self.res = sparse.lil_matrix(A)
            if self.uniform_weights:
                self.res = (self.res != 0).astype(float)
            # end
        # end

        # If the new adj did not come from an appropriate specialized network
        # then adjust the spectral radius.
        if scale_rad:
            self.scale_spect_rad()

        self.res = self.res.tocsr()
    # end

    def dense_init(self, *args, scale_rad=True):
        """ Initialize a sparse reservoir with the adjacency matrix of and Erdos-Renyi
            random graph. Remove self edges. Use uniform weights or integer weights.
            Scale the spectral radius.
        """
        # Create random reservoir
        if len(args) == 2:
            res_sz, connect_p = args
            if self.uniform_weights:
                self.res = (np.random.rand(res_sz, res_sz) < connect_p).astype(float)
            else:
                self.res = np.random.rand(res_sz, res_sz)
                self.res[self.res > connect_p] = 0
                self.res *= 2.0
            # end
        # end

        # Create reservoir from given adj matrix
        if len(args) == 1:
            self.res = args[0]
            if sparse.issparse(self.res):
                self.res = self.res.toarray()
            if self.uniform_weights:
                self.res = (self.res != 0).astype(float)
            # end
        # end

        # If the new adj did not come from an appropriate specialized network
        # then adjust the spectral radius.
        if scale_rad:
            self.scale_spect_rad()
    # end

    def scale_spect_rad(self):
        """ Scales the spectral radius of the reservoir so that spectral_radius(self.res) = self.spect_rad
        """
        sp_res = sparse.issparse(self.res)
        # Remove self edges
        if sp_res:
            self.res = self.res.tolil()
        for i in range(self.res.shape[0]): self.res[i,i] = 0.0

        # Adjust spectral radius
        if sp_res:
            lam = np.linalg.eigvals(self.res.toarray())
        else:
            lam = np.linalg.eigvals(self.res)
        # end
        curr_rad = np.max(np.abs(lam))
        if np.isclose(curr_rad,0, 1e-8):
            raise Exception("Reservoir is too sparse to find spectral radius")
        # end
        self.res *= self.spect_rad/curr_rad

    def drive(self,t,u):
        """
        Parameters
        t (1 dim ndarray): an array of time values
        u (function)     : for each i, u(t[i]) produces the state of the system that is being learned
        """

        # Reservoir ode
        def res_f(r,t):
            return self.gamma*(-1*r + self.activ_f(self.res.dot(r) + self.sigma*self.W_in.dot(u(t))))
        #end

        r_0    = self.state_0
        states = integrate.odeint(res_f,r_0,t)
        self.state_0 = states[-1]
        return states
    # end

    def fit(self, t, u):
        """
        Parameters
        t (1 dim ndarray): an array of time values
        u (function)     : for each i, u(t[i]) produces the state of the system that is being learned
        """
        driven_states    = self.drive(t,u)
        true_states      = u(t).T
        if self.solver == "least squares":
            self.W_out = np.linalg.lstsq(driven_states, true_states, rcond=None)[0].T
        # end
        else:
            ridge_regression = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge_regression.fit(driven_states,true_states)
            self.W_out       = ridge_regression.coef_
        # end
        error = np.mean(np.linalg.norm(self.W_out.dot(driven_states.T)-true_states.T,ord=2,axis=0))
        self.is_trained = True
        return error
    # end


    def predict(self, t, u_0=None, r_0=None, return_states=False):

        if not self.is_trained:
            raise Exception("Reservoir is untrained")

        # Reservoir prediction ode
        def res_pred_f(r,t):
            return self.gamma*(-1*r + self.activ_f(self.res.dot(r) + self.sigma * self.W_in.dot(self.W_out.dot(r))))
        # end
        if r_0 is None and u_0 is None:
            r_0  = self.state_0
        # end
        elif r_0 is None and u_0 is not None:
            r_0 = self.W_in.dot(u_0)
        # end
        pred = integrate.odeint(res_pred_f, r_0, t)
        if return_states:
            return self.W_out.dot(pred.T), pred.T
        return self.W_out.dot(pred.T)
    # end

    def specialize(self, base, random_W_in=True):
        """ Specializes the reservoir nodes. Reinitializes W_in, W_out and the initial reservoir
            state. Network specialization  outlined in:
                Spectral and Dynamic Consequences of Network Specialization
                https://arxiv.org/abs/1908.04435

            Parameters
            base (list of integers): nodes in the graph that should remain fixed
            random_W_in (bool): Create a new random W_in if true, copy rows from the old
                                W_in so that specialized nodes receive the same input as
                                their ancestor node in the original reservoir

            Notes:
            Reservoir needs to be refitted to the data after specialization
        """
        # Check for non negative entries
        scale_rad = np.sum(self.res < 0) != 0

        S, origin = specialize(self.res, base)
        if self.sparse_res:
            self.sparse_init(S, scale_rad=scale_rad)
        else:
            self.dense_init(S, scale_rad=scale_rad)
        # end

        # Reinitialize reservoir
        res_sz  = S.shape[0]
        num_in  = self.W_in.shape[1]
        num_out = self.W_out.shape[0]
        self.is_trained  = False
        self.W_out       = np.zeros((num_out, res_sz))
        self.state_0     = np.random.rand(res_sz)

        if random_W_in:
            self.W_in        = np.random.rand(res_sz, num_in) - .5
        else:

            new_W_in = np.zeros((res_sz,num_in))
            for i in range(res_sz):
                # Copy selected rows of the old W_in mapping that correspond with node origins
                new_W_in[i,:] = self.W_in[origin[i],:]
            # end
            self.W_in = new_W_in
        # end
    # end

    def score_nodes(self, t, u, r_0=None, u_0=None):
        """ Give every node in the reservoir a relative importance score based on the pruning metric
            outlined in:
                Pruning Convolutional Neural Networks for Resource Efficient Inference
                https://arxiv.org/abs/1611.06440

            Parameters
            ----------
            u  (solve_ivp solution): system to model
            t  (ndarray): time values to test

            Returns
            -------
            scores (ndarray): Each node's importance score
        """
        if not self.is_trained:
            raise Exception("Reservoir is untrained")

        pre, r     = self.predict(t, return_states=True, r_0=r_0, u_0=u_0)
        derivative = self.W_out.T.dot(pre - u(t))
        scores     = np.mean(np.abs(derivative*r), axis=1)
        return scores
    # end

# end class
