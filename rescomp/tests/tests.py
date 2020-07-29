from rescomp import *
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

ORIG_RES = {
    "res_sz": 30,
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

LORENZ = {
    "x0": [-20, 10, -.5],
    "begin": 0,
    "end": 60,
    "timesteps":60000,
    "train_per": .66,
    "solver": lorenz_equ
}

RES = {
    "res_sz": 50,
    "activ_f": np.tanh,
    "connect_p": .12,
    "ridge_alpha": .0001,
    "spect_rad": 0.5,
    "gamma": 5.,
    "sigma": 1.5,
    "uniform_weights": True,
    "solver": "ridge",
    "sparse_res": True,
    "signal_dim": 2
}

DRIVEN = {
    "delta": 0.01,
    "drive_dim": 2
}


BATCH = {
    "batchsize": 21,
}


def test_rc():
    params = deepcopy(ORIG_RES)
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
    param_copy = deepcopy(ORIG_RES)
    param_copy["res_sz"] = rc.res.shape[0]
    param_copy["connect_p"] = np.sum(rc.res != 0)/ (rc.res.shape[0]**2)
    rc_ctrl = ResComp(**param_copy)
    assert rc.res.shape[0] == rc_ctrl.res.shape[0]

def test_ResComp_init():
    params = deepcopy(ORIG_RES)
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
    params = deepcopy(ORIG_RES)
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

def test_spect_rad():
    rc = ResComp()
    for i in range(10):
        A = rc.random_graph(200).toarray()
        scc_rho = rc._spectral_rad(A)
        iter_rho = np.max(np.abs(np.linalg.eigvals(A)))
        assert np.isclose(scc_rho, iter_rho)

def test_zero_radius():
    A = np.zeros((200,200))
    rc = ResComp(A, sparse_res=True)
    assert rc.spect_rad == 0.0

# Batch and Driven Reservoir Computers

# # Test initialization

def init_correct(model, params):
    rc = model(**params)
    for k in params.keys():
        msg = f"Key {k} evaluates differently in {model}, ({rc.__dict__[k]}) and params ({params[k]})"
        if k in ["connect_p", "spect_rad"]:
            assert np.abs(rc.__dict__[k] - params[k]) <= .01
        else:
            assert rc.__dict__[k] == params[k], msg
    return True

def test_each_model(f):
    assert f(ResComp, RES)
    assert f(DrivenResComp, {**DRIVEN, **RES})
    assert f(BatchResComp, {**BATCH, **RES})
    assert f(BatchDrivenResComp, {**BATCH, **DRIVEN, **RES})

# # Test driving
def make_test_data():
    tr = np.linspace(0,10, 500)
    ts = np.linspace(10,15, 250)
    drive = lambda x: np.array([np.sin(x), np.cos(x)]).T
    signal = lambda x: np.array([np.cos(x), -1 * np.sin(x)]).T
    return tr, ts, drive, signal


def drive_correct(model, ps):
    driven = model in [DrivenResComp, BatchDrivenResComp]
    tr, ts, drive, signal = make_test_data()
    rc = model(**ps)
    if driven:
        out = rc.drive(tr, drive, signal)
    else:
        out = rc.drive(tr, signal)
    if out is not None:
        m, n = out.shape
        return m == len(tr) and n == rc.res_sz
    else:
        return not np.all(rc.Hbar == 0)

def test_batch_size_too_big():
    ps = {**RES, "batchsize": 255}
    assert drive_correct(BatchResComp, ps)
    assert drive_correct(BatchDrivenResComp, {**ps, **DRIVEN})

# # Test fitting
def fit_correct(model, ps):
    driven = not model in [ResComp, BatchResComp]
    rc = model(**ps)
    tr, ts, drive, signal = make_test_data()
    u = lambda x: signal(x).T # Signal or transposed signal should work
    if driven:
        err = rc.fit(tr, signal, drive)
        err = rc.fit(tr, u, drive)
    else:
        err = rc.fit(tr, signal)
        err = rc.fit(tr, u)
    return True

# # Test Predicitions
def predict_correct(model, ps):
    driven = not model in [ResComp, BatchResComp]
    rc = model(**ps)
    tr, ts, drive, signal = make_test_data()
    rc.state_0 = rc.W_in @ signal(tr[0])
    if driven:
        err = rc.fit(tr, signal, drive)
        pre = rc.predict(tr, drive, u_0=signal(tr[0]))
    else:
        err = rc.fit(tr, signal)
        pre = rc.predict(tr, u_0=signal(tr[0]))
    error = np.mean(np.linalg.norm(pre.T - signal(tr), ord=2, axis=0)**2)**(1/2)
    if error < 1.0:
        return True
    return False

# # Test inverse free
def inv_free_correct():
    models = [BatchResComp, BatchDrivenResComp]
    params = [{**RES, **BATCH}, {**RES, **BATCH, **DRIVEN}]
    tr, ts, drive, signal = make_test_data()
    for model, ps in zip(models, params):
        rc = model(**ps)
        if model == BatchResComp:
            rc.fit(tr, signal, inv_free=True)
            pre = rc.predict(tr, u_0=signal(tr[0]), inv_free=True)
        else:
            rc.fit(tr, signal, drive, inv_free=True)
            pre = rc.predict(tr, drive, u_0=signal(tr[0]), inv_free=True)
        assert not np.all(rc.Hbar == 0)
        assert not np.all(rc.Ybar == 0)
        error = np.mean(np.linalg.norm(pre.T - signal(tr), ord=2, axis=0)**2)**(1/2)
        assert error < 1.0

# # Test partition

def random_time_array(n, start=0):
    t = [start]
    def nextsample(t):
        t[0] += np.random.rand()
        return t[0]
    return [nextsample(t) for i in range(n)]

def uniform_time_array(n, start=0, end=500):
    return np.linspace(start, end, n)


def test_window():
    """ Make sure each partition is smaller than the given time window """
    rc = ResComp(**RES)
    for window in [.5, 3, 1001]:
        for timef in [random_time_array, uniform_time_array]:
            times = timef(1000)
            ts = rc._partition(times, window, 0)
            for sub in ts:
                assert sub[-1] - sub[0] <= window + 1e-12

def test_overlap():
    """ Ensure that overlap is correct on average """
    rc = ResComp(**RES)
    for window in [30, 100]:
        for overlap in [.1, .9,]:
            for timef in [random_time_array, uniform_time_array]:
                times = timef(1000)
                ts = rc._partition(times, window, overlap)
                prev = None
                over = 0.0
                for sub in ts:
                    if prev is not None:
                        inters = set(sub).intersection(set(prev))
                        over += len(inters) / len(sub)
                    prev = sub
                assert np.abs(over/len(ts) - overlap) < .05

# Test ResComp
test_rc()
test_ctrl()
test_init()
test_ResComp_init()
test_topologies()
test_spect_rad()
test_zero_radius()

# Test specialization
test_spec_best()
test_specialize1()
test_specialize2()
test_specialize3()
test_origin()
test_origin2()

# Additional tests for new models
test_each_model(init_correct)
test_each_model(drive_correct)
test_batch_size_too_big()
test_each_model(fit_correct)
test_each_model(predict_correct)
inv_free_correct()
test_overlap()
test_window()

print("All tests passed")
