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
        self.W_in        = np.random.rand(res_sz, num_in) - 1.
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
        self.network     = network
        self.connect_p   = connect_p
        self.res_sz      = res_sz
        self.min_weight  = min_weight
        self.max_weight  = max_weight
        self.uniform_weights = uniform_weights

        # Make reservoir based on number of arguments
        if len(args) == 0:
            A = self.make_reservoir(res_sz, network)
        if len(args) == 1:
            A = sparse.lil_matrix(args[0])
            res_sz = A.shape[0]
        if len(args) == 2:
            A = self.make_reservoir(res_sz, network)
        # end
        
        if not sparse_res:
            A = A.toarray()
        self.res = A
        if self.uniform_weights:
            self.res = (self.res != 0).astype(float)
        # end
        
        # Multiply matrix by a constant to achive the desired spectral radius
        self.scale_spect_rad()
    # end
    
    def make_reservoir(self, res_size, network):
        if network == "preferential attachment":
            A = self.preferential_attachment(res_size)
        elif network == "small world":
            A = self.small_world(res_size)
        elif network == "random graph":
            A = self.random_graph(res_size)
        else:
            raise ValueError(f"The network argument {network} is not in the list [\"preferential attachment\", \"small world\", \"random graph\"]")
        # end
        return A
    
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
        return sparse.random(n,n, density=self.connect_p, dtype=float, format="lil", data_rvs=self.weights)
    
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
        derivative = self.W_out.T.dot(pre - u(t))
        scores     = np.mean(np.abs(derivative*r), axis=1)
        return scores
    # end

# end class
