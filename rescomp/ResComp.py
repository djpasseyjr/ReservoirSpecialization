import numpy as np
from sklearn.linear_model import Ridge
from scipy import integrate
from scipy import sparse
from rescomp.specialize import *
from math import floor, ceil
from scipy import integrate
from warnings import warn


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

        Keyword Arguments:

        res_sz:          (Int) Number of nodes in reservoir
        signal_dim:      (Int) Dimension of the training signal

        network:         (String) Reservoir network topology. Choose from ["random graph", "preferential attachment", "small world"]
        connect_p:       (Float) Edge probability used if network="random graph"
        spect_rad:       (Float) Desired reservoir spectral radius
        sigma:           (Float) Reservoir ode hyperparameter
        gamma:           (Float) Reservoir ode hyperparameter
        solver:          (String) Specify solver. Options = ["least squares", "ridge"].
        ridge_alpha:     (Float) Regularization parameter for the ridge regression solver
        activ_f:         (Function) Activation function for reservoir nodes. Used in ODE
        sparse_res:      (Bool) Chose to use sparse matrixes or dense matrixes
        uniform_weights: (Bool) Choose between uniform or random edge weights
        max_weight:      (Float) Maximim edge weight if uniform_weights=False
        min_weight:      (Float) Minimum edge weight if uniform_weights=False.
                        ** Note that all weights are scaled after initialization
                           to achive desired spectral radius **
    """
    def __init__(self, *args,
                 res_sz=200,   activ_f=np.tanh,
                 connect_p=.03, ridge_alpha=.00001,
                 spect_rad=.9, sparse_res=True,
                 sigma=0.1,    uniform_weights=True,
                 gamma=1.,     solver="ridge regression",
                 signal_dim=3, network="random graph",
                 max_weight=2, min_weight=0
                ):

        # Support for depriciated version
        num_in = signal_dim
        num_out = signal_dim
        if len(args) == 2:
            num_in, num_out = args
        # end

        # Set model attributes
        self.signal_dim  = signal_dim
        self.gamma       = gamma
        self.sigma       = sigma
        self.activ_f     = activ_f
        self.ridge_alpha = ridge_alpha
        self.state_0     = np.random.rand(res_sz)
        self.solver      = solver
        self.sparse_res  = sparse_res
        self.spect_rad   = spect_rad
        self.is_trained  = False
        self.network     = network
        self.connect_p   = connect_p
        self.res_sz      = res_sz
        self.min_weight  = min_weight
        self.max_weight  = max_weight
        self.uniform_weights = uniform_weights
        self.time_dim = None

        # Make reservoir based on number of arguments
        if len(args) == 0:
            A = self.make_reservoir(res_sz, network)
        if len(args) == 1:
            self.network = "unknown topology"
            if self.sparse_res:
                A = sparse.lil_matrix(args[0])
            else:
                A = args[0]
            res_sz = A.shape[0]
        if len(args) == 2:
            A = self.make_reservoir(res_sz, network)
        # end

        if not sparse_res and sparse.issparse(A):
            A = A.toarray()

        self.res = A
        if self.uniform_weights:
            self.res = (self.res != 0).astype(float)
        # end

        # Multiply matrix by a constant to achive the desired spectral radius
        self.scale_spect_rad()
        # Adjust data members to match reservoir
        self.set_res_data_members()
    # end

    def make_reservoir(self, res_size, network):
        if network == "preferential attachment":
            A = self.preferential_attachment(res_size)
        elif network == "small world":
            A = self.small_world(res_size)
        elif network == "random graph":
            A = self.random_graph(res_size)
        else:
            raise ValueError(f"The network argument \"{network}\" is not in the list [\"preferential attachment\", \"small world\", \"random graph\"]")
        # end
        return A

    def set_res_data_members(self):
        self.res_sz = self.res.shape[0]
        self.connect_p = np.sum(self.res != 0)/(self.res_sz)**2
        self.W_in        = np.random.rand(self.res_sz, self.signal_dim) - 0.5
        self.W_out       = np.zeros((self.signal_dim, self.res_sz))
        self.state_0     = np.random.rand(self.res_sz)
        self.spect_rad = self._spectral_rad(self.res)
        # Determine the max and min edge weights
        if self.sparse_res:
            edge_weights = list(sparse.dok_matrix(self.res).values())
        else:
            edge_weights = self.res[self.res != 0]
        if len(edge_weights) == 0:
            self.max_weight = 0
            self.min_weight = 0
        else:
            self.max_weight = np.max(edge_weights)
            self.min_weight = np.min(edge_weights)
        self.uniform_weights = (self.max_weight - self.min_weight) < 1e-12

    def _spectral_rad(self, A):
        """ Compute spectral radius via max radius of the strongly connected components """
        g = nx.DiGraph(A.T)
        scc = nx.strongly_connected_components(g)
        rad = 0
        for cmp in scc:
            # If the component is one node, spectral radius is the edge weight of it's self loop
            if len(cmp) == 1:
                i = cmp.pop()
                max_eig = A[i,i]
            else:
                # Compute spectral radius of strongly connected components
                adj = nx.adj_matrix(nx.subgraph(g,cmp))
                max_eig = np.max(np.abs(np.linalg.eigvals(adj.T.toarray())))
            if max_eig > rad:
                rad = max_eig
        return rad

    def scale_spect_rad(self):
        """ Scales the spectral radius of the reservoir so that
            _spectral_rad(self.res) = self.spect_rad
        """
        curr_rad = self._spectral_rad(self.res)
        if not np.isclose(curr_rad,0, 1e-8):
            self.res *= self.spect_rad/curr_rad
        else:
            warn("Spectral radius of reservoir is close to zero. Edge weights will not be scaled")
        # end
        # Convert to csr if sparse
        if sparse.issparse(self.res):
            self.res = self.res.tocsr()


    #-------------------------------------
    # Graph topology options
    #-------------------------------------

    def weights(self,n):
        if self.uniform_weights:
            return np.ones(n)
        else:
            return (self.max_weight-self.min_weight)*np.random.rand(n) + self.min_weight

    def random_graph(self, n):
        """ Create the sparse adj matrix of a random directed graph
            on n nodes with probability of any link equal to connect_p
        """
        A = sparse.random(n,n, density=self.connect_p, dtype=float, format="lil", data_rvs=self.weights)
        # Remove self edges
        for i in range(n):
             A[i,i] = 0.0
        return A

    def preferential_attachment(self, n, m=2):
        """ Create a network via preferential attachment
        """
        B = nx.barabasi_albert_graph(n,m)
        A = nx.adj_matrix(self._randomly_direct(B)).T
        A = A.astype(float)
        A[A != 0] = self.weights(np.sum(A != 0))
        return A

    def small_world(self, n, k=5, p=.05):
        """ Create a small world network. (Watts-Strogatz model)
        """
        S = nx.watts_strogatz_graph(n,k,p)
        A = nx.adj_matrix(self._randomly_direct(S)).T
        A = A.astype(float)
        A[A != 0] = self.weights(np.sum(A != 0))
        return A

    def _randomly_direct(self, G):
        """ Helper function to randomly direct undirected networkx graphs.
            Accepts undirected graphs, directs the edges then randomly
            deletes half of the directed edges.
        """
        edges = list(G.to_directed().edges)
        M = len(edges)
        idxs = np.random.choice(range(M),size=M//2, replace=False).astype(int)
        new_edges = [edges[i] for i in idxs]
        direct = nx.DiGraph()
        direct.add_nodes_from(range(len(G.nodes)))
        direct.add_edges_from(new_edges)
        return direct

    #---------------------------
    # Train and Predict
    #---------------------------
    def time_dimension(self, t, u):
        """ Determine if the size of the first or second dimension of u(t) varies with t """
        if len(t) <= 1:
            return None
        m2, n2 = u(t[:2]).shape
        test_t = np.array([t[0], (t[0] + t[1]) / 2,  t[1]])
        m3, n3 = u(test_t).shape
        if m2 != m3:
            self.time_dim = 0
        else:
            self.time_dim = 1

    def time_dim_0(self, t, u, state_array):
        """ Ensure that u(t) produces a (n timesteps) x (signal dimension) array """
        self.time_dimension(t, u)
        if self.time_dim is None:
            return state_array
        elif self.time_dim == 1:
            return state_array.T
        return state_array

    def drive(self,t,u):
        """
        Parameters
        t (1 dim ndarray): an array of time values
        u (function)     : for each i, u(t[i]) produces the state of the system that is being learned
        """

        # Reservoir ode
        def res_f(r,t):
            return self.gamma * (-1 * r + self.activ_f(self.res @ r + self.sigma * self.W_in @ u(t)))
        #end

        r_0    = self.state_0
        states = integrate.odeint(res_f, r_0, t)
        self.state_0 = states[-1]
        return states
    # end

    def solve_wout(self, internal_states, targets):
        if self.solver == "least squares":
            W_out = np.linalg.lstsq(internal_states, targets, rcond=None)[0].T
        else:
            ridge_regression = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge_regression.fit(internal_states, targets)
            W_out       = ridge_regression.coef_
        return W_out

    def fit(self, t, u, return_states=False):
        """
        Parameters
        t (1 dim ndarray)    : an array of time values
        u (function)         : for each i, u(t[i]) produces the state of the system that is being learned
        return_states (bool) : If True returns the node states of the reservoir

        Returns
        err (float) : Error in the fit (2-norm of residuals)
            Optionally returns: drive_states : (ndarray) of node states
        """
        driven_states = self.drive(t,u)
        true_states = u(t)
        true_states = self.time_dim_0(t, u, true_states) # Ensure number of columns == signal dimension
        self.W_out = self.solve_wout(driven_states, true_states)
        self.is_trained = True
        # Compute error
        diff = self.W_out @ driven_states.T - true_states.T
        error = np.mean(np.linalg.norm(diff, ord=2, axis=0)**2)**(1/2)
        if return_states:
            # Return node states
            return error, driven_states
        return error
    # end

    def fit_batch(self, t, u, time_window, overlap=0.0, return_states=False):
        """ Train reservoir computer on the signal as a batch of short signals instead
            of on one long continuous signal

            Parameters
            t (1 dimensional array or list of one dimensional arrays): Time values corresponding to signal u
            u (function or list of functions that accept time values): Input signal
            time_window (float): How long to make each shorter signal in time units. (If t is in seconds
                setting time_window to 2.0 will mean that each shorter signal is 2 seconds long
            overlap (float): Less than one, greater or equal to zero. (Defaults to zero) Percent that
                each signal window overlaps the previous signal window
        """
        internal_states, target = self.drive_batch(t, u, time_window, overlap=overlap)
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

    def drive_batch(self, t, u, time_window, overlap=0.0):
        """ Convert signal to a batch of signals and drive the reservoir with each one.
            Concatenate the output signals and return as one array each of states and targets.
        """
        internals = ()
        targets = ()
        # Dispatch on types of u and t
        if isinstance(u, list) and isinstance(t, list):
            for time, signal in zip(t, u):
                internal, target = self._drive_batch(time, signal, time_window, overlap=overlap)
                internals += internal # Concatenate internals
                targets += target # Concatenate targets
        else:
            internals, targets = self._drive_batch(t, u, time_window, overlap=overlap)
        # Stack internal states and targets
        internals = np.vstack(internals)
        targets = np.vstack(targets)
        return internals, targets

    def _drive_batch(self, t, u, time_window, overlap=0.0):
        ts = self._partition(t, time_window, overlap=overlap)
        internals = ()
        targets = ()
        for time in ts:
            assert (time[-1] - time[0]) < time_window, f"Subarray covers {time[-1] - time[0]} seconds but `time_window` is {time_window}"
            # Set initial condition
            self.state_0 = self.W_in @ u(time[0])
            internals += (self.drive(time,u),)
            targets += (u(time).T,)
        return internals, targets

    def _partition(self, t, time_window, overlap=0.0):
        """ Partition `t` into subarrays that each include `time_window` seconds. The variable
            `overlap` determines what percent of each sub-array overlaps the previous sub-array.
            The last subarray may not contain a full time window.
        """
        if (overlap >= 1) or (overlap < 0.0):
            raise ValueError("Overlap argument must be greater than or equal to zero and less than one")

        ts = ()
        start = 0
        tmax = t[start] + time_window
        for i,time in enumerate(t):
            while time > tmax:
                end = i
                if end - start == 1:
                    warn("rescomp.ResComp._partition partitioning time array into single entry arrays. Consider increasing time window")
                ts += (t[start:end],)
                diff = floor((end - start) * (1.0 - overlap))
                start += max(diff, 1)
                tmax = t[start] + time_window
        ts += (t[start:],)
        return ts


    def predict(self, t, u_0=None, r_0=None, return_states=False):
        """
            Parameters:
            -----------
        """
        if not self.is_trained:
            raise Exception("Reservoir is untrained")

        # Reservoir prediction ode
        def res_pred_f(r,t):
            return self.gamma*(-1*r + self.activ_f(self.res @ r + self.sigma * self.W_in @ (self.W_out @ r)))
        # end
        if r_0 is None and u_0 is None:
            r_0  = self.state_0
        # end
        elif r_0 is None and u_0 is not None:
            r_0 = self.W_in @ u_0
        # end
        pred = integrate.odeint(res_pred_f, r_0, t)
        if return_states:
            return self.W_out @ pred.T, pred.T
        return self.W_out @ pred.T
    # end


    #---------------------------------------------
    # Specialization and node importance ranking
    #---------------------------------------------

    def specialize(self, base, random_W_in=True):
        """ Specializes the reservoir nodes. Reinitializes W_in, W_out and the initial reservoir
            state. Network specialization  outlined in:
                Spectral and Dynamic Consequences of Network Specialization
                https://arxiv.org/abs/1908.04435

            Parameters:
            base (list of integers): nodes in the graph that should remain fixed
            random_W_in (bool): Create a new random W_in if true, copy rows from the old
                                W_in so that specialized nodes receive the same input as
                                their ancestor node in the original reservoir

            Notes:
            Reservoir needs to be refitted to the data after specialization
        """
        # Check for non negative entries

        S, origin = specialize(self.res, base)
        self.res = S
        self.res_sz = S.shape[0]

        # Check if the matrix is non-negative
        if np.sum(self.res < 0) != 0:
            # If not, we need to scale the spectral radius
            self.scale_spect_rad()

        # Reinitialize reservoir
        res_sz  = S.shape[0]
        num_in  = self.W_in.shape[1]
        num_out = self.W_out.shape[0]
        self.is_trained  = False
        self.W_out       = np.zeros((num_out, res_sz))
        self.state_0     = np.random.rand(res_sz)

        if random_W_in:
            self.W_in        = np.random.rand(res_sz, num_in) - 1.
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
            u  (solve_ivp solution): system to model
            t  (ndarray): time values to test

            Returns
            scores (ndarray): Each node's importance score
        """
        if not self.is_trained:
            raise Exception("Reservoir is untrained")

        pre, r     = self.predict(t, return_states=True, r_0=r_0, u_0=u_0)
        derivative = self.W_out.T @ (pre - u(t))
        scores     = np.mean(np.abs(derivative*r), axis=1)
        return scores
    # end

# end class



#------------------------------------------------------------
# Example System: Lorenz Equations
#------------------------------------------------------------
def lorenz_deriv(t0, X, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorenz system."""
    (x, y, z) = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
# end

def lorenz_equ(x0=[-20, 10, -.5], begin=0, end=60, timesteps=60000, train_per=.66):
    """Use solve_ivp to produce a solution to the lorenz equations"""
    t = np.linspace(begin,end,timesteps)
    n_train = floor(train_per*len(t))
    train_t = t[:n_train]
    test_t = t[(n_train+1):]
    u = integrate.solve_ivp(lorenz_deriv, (begin,end),x0, dense_output=True).sol
    return train_t, test_t, u
