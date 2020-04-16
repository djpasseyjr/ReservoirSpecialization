from ResComp import *
from specializeGraph import *
from math import floor
from scipy import integrate

def lorentz_deriv(t0, X, sigma=10., beta=8./3, rho=28.0):
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
    u = integrate.solve_ivp(lorentz_deriv, (begin,end),x0, dense_output=True).sol
    return train_t, test_t, u

diff_eq_params = {"x0": [-20, 10, -.5],
                  "begin": 0,
                  "end": 60,
                  "timesteps":60000,
                  "train_per": .66,
                  "solver": lorenz_equ
                 }

RES_PARAMS = {"res_sz": 30,
              "activ_f": np.tanh,
              "connect_p": .12,
              "ridge_alpha": .0001,
              "spect_rad": .9,
              "gamma": 1.,
              "sigma": 0.12,
              "uniform_weights": True,
              "solver": "ridge",
              "sparse_res": True,
              "signal_dim": 3
              }


def test_rc():
    params = deepcopy(RES_PARAMS)
    params["solver"] = "ridge"
    params["ridge_alpha"] = 0.00001
    params["uniform_weights"] = True
    params["sparse_res"] = False
    A = np.array([[0,1,0, 1,0,0,0,1],
                    [0,0,1, 0,1,0,1,0],
                    [1,0,0, 0,0,1,0,0],

                    [0,0,1, 0,0,0,0,0],
                    [0,1,0, 0,0,0,0,0],
                    [1,0,0, 1,0,0,0,0],
                    [0,1,1, 0,1,0,0,0],
                    [0,1,0, 0,0,1,0,0],
                   ]).astype(float)
    W_in = np.ones((8,3))
    R = np.random.rand(8,8)
    R[A == 0] = 0
    R = sparse.csr_matrix(R)
    rc = ResComp(R,**params)
    rc.state_0 = np.ones(8)
    rc.W_in = W_in
    return rc


def test_spec_best():
    rc = test_rc()
    r_0 = rc.state_0
    train_t, test_t, u = lorenz_equ()
    num_nodes=3
    rc.fit(train_t,u)
    scores = rc.score_nodes(train_t, u, r_0=r_0)
    worst_nodes = np.argsort(scores)[:-num_nodes]
    rc.specialize(worst_nodes, random_W_in=False)
    assert rc.res.shape[0] == 49

def test_ctrl():
    rc = test_rc()
    r_0 = rc.state_0
    train_t, test_t, u = lorenz_equ()
    num_nodes=3
    rc.fit(train_t,u)
    scores = rc.score_nodes(train_t, u, r_0=r_0)
    worst_nodes = np.argsort(scores)[:-num_nodes]
    rc.specialize(worst_nodes, random_W_in=False)
    param_copy = deepcopy(RES_PARAMS)
    param_copy["res_sz"] = rc.res.shape[0]
    param_copy["connect_p"] = np.sum(rc.res != 0)/ (rc.res.shape[0]**2)
    rc_ctrl = ResComp(**param_copy)
    assert rc.res.shape[0] == rc_ctrl.res.shape[0]
    
def test_ResComp_init():
    params = deepcopy(RES_PARAMS)
    params["uniform_weights"] = True
    A = make_path_matrix()
    sp_rc = ResComp(A, **params)
    dense_params = deepcopy(params)
    dense_params["sparse_res"] = False
    d_rc = ResComp(A, **dense_params)
    
    assert sparse.issparse(sp_rc.res)
    assert not sparse.issparse(d_rc.res)
    assert np.all(sp_rc.res.toarray() == d_rc.res)
    # No Argument Initialization
    assert sparse.issparse(ResComp(**params).res)
    assert not sparse.issparse(ResComp(**dense_params).res)
    
def test_topologies():
    params = deepcopy(RES_PARAMS)
    params["network"] = "preferential attachment"
    rc = ResComp(**params)
    assert np.sum(rc.res != 0) == 2*rc.res_sz - 4
    
    params["network"] = "small world"
    rc = ResComp(**params)
    assert np.sum(rc.res != 0) == rc.res_sz*2
    

def make_matrix():
     return np.array([[0,1,0, 1,0,0,0,1],
                      [0,0,1, 0,1,0,1,0],
                      [1,0,0, 0,0,1,0,0],

                      [0,0,1, 0,0,0,0,0],
                      [0,1,0, 0,0,0,0,0],
                      [1,0,0, 1,0,0,0,0],
                      [0,1,1, 0,1,0,0,0],
                      [0,1,0, 0,0,1,0,0],
                     ])

def make_path_matrix():
     return np.array([[0,1,1, 0,0,0,0,1],
                      [0,0,1, 0,0,1,0,1],
                      [1,0,0, 0,0,0,0,0],

                      [1,0,1, 0,0,0,0,0],
                      [0,0,0, 0,0,0,0,0],
                      [0,0,0, 1,0,0,0,0],
                      [0,0,0, 0,1,0,0,0],
                      [0,0,0, 0,0,1,0,0],
                     ])


def test_init():
    A = np.random.rand(100,100)
    A[A > .01] = 0
    assert np.all(LightGraph(A).adj() == A)
    assert np.all(LightGraph(A).adj() == LightGraph(nx.DiGraph(A.T)).adj())


def test_specialize1():
    C = make_path_matrix()
    lg = LightGraph(C)
    base = [0,1,2]
    G = nx.DiGraph(specializeGraph(C,base).T)
    g, origin = specialize(C,base)
    assert nx.is_isomorphic(G,nx.DiGraph(g.T))

def test_specialize2():
    for i in range(25):
        n = 4
        B = (np.random.rand(n,n) < .3).astype(int)
        base = np.random.choice(range(n), size=1, replace=False)
        for j in range(n): B[j,j] = 0
        G = nx.DiGraph(specializeGraph(B,base).T)
        g, origin = specialize(B,base)
        assert nx.is_isomorphic(G,nx.DiGraph(g.T))

def test_specialize3():
    for i in range(100,200,25):
        B = (np.random.rand(i,i) < .002).astype(int)
        base = np.random.choice(range(i), size=i-(i//10), replace=False)
        for j in range(i): B[j,j] = 0
        G = nx.DiGraph(specializeGraph(B,base).T)

        g, origin = specialize(B,base)
        assert nx.is_isomorphic(G,nx.DiGraph(g.T))

def test_origin():
    n = 20
    base = [0,1,2,3,4]
    p = .1
    # Make a matrix with single node components outside of base set
    # and no  connections between components outside of the base
    A = (np.random.rand(n,n) < p).astype(float)
    A[5:,5:] = np.zeros((n-5,n-5))


    # These nodes should be copied in-degree*out-degree times
    # We check that this is true.
    g = nx.DiGraph(A.T)
    in_dict = g.in_degree()
    in_degree = np.array([in_dict[n] for n in range(A.shape[0]) ])
    out_dict = g.out_degree()
    out_degree = np.array([out_dict[n] for n in range(A.shape[0])])
    pred_copies = in_degree * out_degree

    S, origin = specialize(A, base)
    N = S.shape[0]

    # Count the number of copies of each node
    copies = np.zeros(n)
    for i in range(N):
        copies[origin[i]] += 1
    assert np.all(copies[6:] == pred_copies[6:])

def test_origin2():
    A = np.array([[0,1,1,0,1,0],
                      [1,0,1,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,1,0,0,1],
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0]])
    S, origin = specialize(A, [0])
    N = S.shape[0]
    # Count the number of copies of each node
    copies = np.zeros(6)
    for i in range(N):
        copies[origin[i]] += 1
    assert np.all(copies == np.array([1., 3., 3., 1., 1., 1.]))

test_rc()
test_spec_best()
test_ctrl()
test_ResComp_init()
test_topologies()

test_init()
test_specialize1()
test_specialize2()
test_specialize3()
test_origin()
test_origin2()
