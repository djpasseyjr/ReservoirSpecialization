# ReservoirSpecialization
Reservoir computer class with functionality to specialize and copy the top preforming subnetworks.

## Files
1. `ResComp.py` : The reservoir computer class
2. `specialize.py` : The optimized specialization algorithm
3. `lorenz_sol.py` : The lorenz equations and a solver
4. `specializeGraph` : A depreciated version of the specialization algorithm used for testing the new version
5. `res_comp_tests.py` : Tests

## Examples

#### Learning the lorenz equations

``` from ResComp import *
    from lorenz_sol import *
    
    train_t, test_t, u = lorenz_equ()
    rc = ResComp()
    err = rc.fit(train_t, u) # Returns norm of the residual
    pred = rc.predict(test_t) # Returns the predicted orbit
    
```

#### Specializing the best nodes

``` from ResComp import *
    from lorenz_sol import *
    
    train_t, test_t, u = lorenz_equ()
    rc = ResComp()
    err = rc.fit(train_t, u)
    
    # Specialize the three most useful nodes
    
    node_scores = rc.score_nodes(train_t,u)
    
    # Base is everything except the best 3
    
    base = np.argsort(node_scores)[:-3]
    rc.specialize(base)
    spec_err = rc.fit(train_t, u)
```
