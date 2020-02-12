import numpy as np
import networkx as nx
from scipy.sparse import dok_matrix
from scipy.linalg import block_diag
import itertools
from matplotlib import pyplot as plt

def specializeGraph(A,Base):
    """
    Function to compute the specialization of a graph. Base nodes and
    links between the base nodes remain the same. The remaining nodes
    are specialized.

    Parameters
    ----------
    A (nxn ndarray): Adjacency matrix for a simple directed graph
    Base (list): base nodes (not to be specialized) zero indexed

    Returns
    -------
    S (pxp ndarray): Specialized adjacency matrix

    """
    Base = list(Base) # Base must be a list
    if np.diag(A).sum() != 0:
        raise ValueError("Some vertices have self edges")

    n = A.shape[0]


    #Permute A so that the base nodes come first
    A = baseFirst(A,Base)
    bSize = len(Base)

    #Begin creating the block diagonal of specialized matrix
    B = A[:bSize,:bSize]
    diag = [B]
    links = []
    #Find connected components and compress graph
    smA,comp = compressGraph(A,bSize)

    #Find all paths from a base node to a base node
    #through the connected components
    pressedPaths = findPathsToBase(smA,bSize)

    #For each compressed path find all combinations
    #of nodes that pass through the associated components
    nNodes = bSize
    for Path in pressedPaths:
        compnts = [comp[c] for c in Path]
        paths = pathCombinations(A,compnts)
        #Select all components not in the base node
        compnToAdd = [A[c,:][:,c] for c in compnts[1:-1]]
        for p in paths:
            diag += compnToAdd
            links += linkAdder(p,nNodes,compnts)
            nNodes += sum(map(len,compnToAdd))

    S = block_diag(*diag)
    for l in links: S[l] = 1
    return S

def baseFirst(A,Base):
    """ Permutes the verices of adjacency matrix A so that the nodes
        belonging to the base set come first.
        Parameters
        ----------
        A (nxn ndarray): Adjacency matrix for a simple directed graph
        Base (list): base nodes zero indexed

        Returns
        -------
        pA (pxp ndarray): Permuted adjacency matrix
        """
    m,n = A.shape

    #Find nodes not in the base
    toSpecialize = list(set(range(n)).difference(set(Base)))

    #Permute A so that the base nodes come first
    permute = Base+toSpecialize
    pA = A[permute,:]
    pA = pA[:,permute]
    return pA

def compressGraph(A,bSize):
    """ Creates a new compressed adjacency matrix, smallA, by representing
        each strongly connected component as a single node (if the component
        has no nodes in the base set)

        Parameters
        ----------
        A (nxn array): adjacency matrix with nodes 0 through (bSize-1)
        belonging to the base set

        bSize (int): number of nodes in the base set

        Returns
        -------
        smallA (NxN array): compressed adjacency matrix
        """
    #Find all strongly connected components in the specialization set
    Spec = A[bSize:,bSize:]
    specG = nx.DiGraph(Spec.T)
    SCComp = [np.array(list(c)) for c in nx.strongly_connected_components(specG)]
    numComp = len(SCComp) #How many strongly connected components
    N = bSize+numComp #Size of the compressed graph

    #Make a dictionary of base nodes and component nodes
    comp = {}
    for i in range(bSize):
        comp[i] = np.array([i])

    for i in range(numComp):
        comp[i+bSize] = SCComp[i] + bSize

    #Create a compressed version of A where each strongly connected
    #component is represented by one vertex
    smallA = np.zeros((N,N))
    smallA[:bSize,:bSize] = A[:bSize,:bSize]
    #Find links between base nodes and components
    for i in range(bSize,N):
        smallA[:bSize,i] = (A[:bSize,comp[i]].sum(axis=1)!=0)*1.
        smallA[i,:bSize] = (A[comp[i],:bSize].sum(axis=0)!=0)*1.
        #Find connections between components
        for j in range(bSize,i):
            smallA[i,j] = 1.*( not (A[comp[i],:][:,comp[j]]==0).all())
            smallA[j,i] = 1.*( not (A[comp[j],:][:,comp[i]]==0).all())

    return smallA, comp

def findPathsToBase(A,bSize):
    """ Finds paths between base nodes that travel through the
        specialization set in the compressed graph.

        Parameters
        ----------
        smallA (NxN array): compressed adjacency matrix with nodes
        0 through (bSize-1) belonging to the base

        bSize (int): size of the base set

        Returns
        -------
        pressedPaths (List of ndarrays): all appropriate paths through the
        compressed graph
    """
    M,N = A.shape
    pressedPaths = []

    #For every two nodes in the base find all paths between them
    for b1 in range(bSize):
        for b2 in range(bSize):
            #Remove all other base nodes from the graph so that
            #we only find paths that go through the specialization set
            if b1 == b2:
                #In this case we are looking for a cycle.
                mask = [b1]+list(range(bSize,N))
                newSize = len(mask) + 1
                reduA = np.zeros((newSize,newSize))
                #Because the networkx cycle finders don't do what we need
                #them to do, we create a new graph and find paths instead
                reduA[:-1,:-1] = A[mask,:][:,mask]
                #Remove ingoing edges from the base node and add to new node
                reduA[-1,:] = reduA[0,:]
                reduA[0,:] =  np.zeros(newSize)
                G = nx.DiGraph(reduA.T)
                #Find paths from the base node to the new node
                #same as finding all the cycles
                paths = list(nx.all_simple_paths(G,0,newSize-1))

            else:
                mask = [b1,b2]+list(range(bSize,N))
                reduA = A[mask,:][:,mask]
                #Remove base node interactions
                reduA[:2,:2] = np.zeros((2,2))
                G = nx.DiGraph(reduA.T)
                paths = list(nx.all_simple_paths(G,0,1))

            #Process Paths so that they make sense when the rest of the base
            #set is added to the graph
            for p in paths:
                if p != []:
                    if b1 == b2:
                        p = np.array(p) + bSize-1
                    else:
                        p = np.array(p) + bSize-2
                    p[[0,-1]] = [b1, b2]
                    pressedPaths.append(p)

    return pressedPaths

def pathCombinations(A,compnts):
    """ Given a path through the connected components of A, find every
        unique combination of edges between the components that can be
        followed to complete the given path

        Parameters
        ----------
        A (nxn array): Adjacency matrix of a graph

        compnts (list): list of lists of the nodes in each component that
        the path traverses. Begins and ends with a base node

        Returns
        -------
        allPaths (list of lists): All viable paths corresponding to
        the component list
    """
    linkOpt = []
    pSize = len(compnts)
    #Variable to keep track of number of nodes in the branch
    nNodes = 1

    #Find the links between each adjacent component in the path
    for i in range(pSize-1):
        rows,cols = np.where(A[compnts[i+1],:][:,compnts[i]]==1)
        if i == 0:
            cols += compnts[0]
            rows += nNodes
        elif i == pSize-2:
            rows += compnts[-1]
            cols += nNodes - len(compnts[i])
        else:
            rows += nNodes
            cols += nNodes - len(compnts[i])
        edges = zip(rows,cols)
        nNodes += len(compnts[i+1])
        linkOpt.append(edges)

    allPaths = [list(P) for P in itertools.product(*linkOpt)]
    return allPaths


def linkAdder(path,nNodes,compnts):
    """
    Produces the links needed to add a branch of strongly connected
    components to a graph with nNodes

    Parameters
    ----------
    path (list of tuples): edges between component nodes
    nNodes (int): number of nodes in the original graph
    compnts (list of lists): list of lists of component nodes

    Returns
    -------
    links (list of tuples): links that correspond with adding
    a branch of connected components to the graph
    """
    links = []
    lenP = len(path)
    #TODO: Edge weights, loops

    for i in range(lenP):
        if i == 0:
            links.append((path[i][0]+nNodes-1,path[i][1]))
        elif i == lenP - 1:
            links.append((path[i][0],path[i][1]+nNodes-1))
        else:
            links.append((path[i][0]+nNodes-1,path[i][1]+nNodes-1))

    return links


"""
SPECIALIZE LINKS
"""


def specializeLink(A,bSize,edge,node,comp):

    """ Specializes one link of the adjacency matrix A copying the
        given component

        Parameters
        ----------
        A (nxn numpy array): Adjacency matrix of a simple directed graph
                             with base nodes at indexes 0,1,...,bSize-1
        bSize (int): number of base nodes
        edge (tuple of integers): edge to specialize
        node (integer): which node to copy
        comp (kx1 numpy array): strongly connected component containing node

        Returns
        -------
        sA (NxN numpy array): the new adjacency matrix
    """

    #Return A unchanged if there is no link at edge
    n = A.shape[0]
    if A[edge] == 0:
        return A
    # If node is in the base set, return A unchanged
    if node < bSize:
        return A

    #Determine if the edge is ingoing or outgoing
    #If the edge is contained in the comp, return A unchanged
    outgoing = True
    if edge[0] == node:
        outgoing = False
        if edge[1] in comp:
            return A
    elif edge[0] in comp:
        return A

    #Determine if the link is the only ingoing or outgoing link already
    if outgoing:
        if A[:,comp].sum() - A[comp][:,comp].sum() == 1:
            return A
    else:
        if A[comp,:].sum() - A[comp][:,comp].sum() == 1:
            return A

    #Get component length and which node is used
    CLen = len(comp)
    whichCompNode = np.where(comp==node)[0]

    #Construct new matrix
    sA = block_diag(A,A[comp][:,comp])

    if outgoing:
        newEdge = edge[0],n+whichCompNode
        sA[edge] = 0
        sA[newEdge] = 1
        keptLinks = A[comp]
        keptLinks[:,comp] = np.zeros((CLen,CLen))
        sA[-CLen:,:-CLen] = keptLinks

    else:
        newEdge = n+whichCompNode,edge[1]
        sA[edge] = 0
        sA[newEdge] = 1
        keptLinks = A[:,comp]
        keptLinks[comp,:] = np.zeros((CLen,CLen))
        sA[:-CLen,-CLen:] = keptLinks

    return sA



"""
DELETE SOURCE OR SINK
"""


def delSink(A,bSize,comp=None):
    """ REMOVE SINK COMPONENTS OF GRAPH
        Parameters
        ----------
        A (nxn numpy array): Adjacency matrix of a simple directed graph
                             with base nodes at indexes 0,1,...,bSize-1
        bSize (int): number of base nodes
        comp (kx1 numpy array): strongly connected components
        Returns
        -------
        sA (NxN numpy array): the new adjacency matrix
    """
    if comp is None:
        comp = compressGraph(A,bSize)[1]

    n = A.shape[0]
    numComp = len(comp)
    removed = set()
    compDict = comp.copy()

    #Compute the out degree of each component
    compDeg = np.ones(max(comp.keys())+1)
    for key in compDict.keys():
        if key >= bSize:
            c = compDict[key]
            compDeg[key] = A[:,c].sum() - A[c][:,c].sum()

    #Find each sink
    sinks = np.where(compDeg==0)[0]

    #While there are sink components in the graph
    while sinks.size > 0 :
        N = A.shape[0]

        #Look at the first sink in the list
        sink = sinks[0]

        #Label each node in the sink
        mask = np.zeros(N,dtype=bool)
        mask[comp[sink]] = True

        #Remove all edges pointing to the sink
        A[mask] = 0

        #Add the component to the remove list
        removed.add(sink)
        compDict.pop(sink)
        compDeg[sink] = 1

        #Recompute the in-degree of each component
        for key in compDict.keys():
            if key >= bSize:
                c = compDict[key]
                compDeg[key] = A[:,c].sum() - A[c][:,c].sum()

        #Find any new sinks
        sinks = np.where(compDeg==0)[0]

    #Remove sinks and relabel components
    relabel = dict(zip(np.arange(n),np.arange(n)))
    mask = np.ones(n,bool)

    for key in comp.keys():
        if key in removed:
            mask[comp[key]] = False
            for node in comp[key]:
                for k in range(node+1,n):
                    relabel[k] -= 1

    A = A[mask][:,mask]


    #Relabel the components
    for key in comp.keys():
        nodes = comp[key]
        for i in range(len(nodes)):
            nodes[i] = relabel[nodes[i]]
            comp[key] = nodes
        if key in removed:
            comp.pop(key)

    return A,comp

def delSource(A,bSize,comp):
    """ Remove the source componenets of a graph that are not included
        in the base set

        Parameters
        ----------
        A (nxn numpy array): Adjacency matrix of a simple directed graph
                             with base nodes at indexes 0,1,...,bSize-1
        bSize (int): number of base nodes
        comp (kx1 numpy array): strongly connected components
        Returns
        -------
        sA (NxN numpy array): the new adjacency matrix
    """
    sA,comp = delSink(A.T,bSize,comp)
    return sA.T,comp


"""
SPECIALIZE ALL OUTGOING OR INCOMING EDGES
"""


def outSpecialize(A,base):

    if np.diag(A).sum() != 0:
        raise ValueError("Some vertices have self edges")

    n = A.shape[0]

    #Permute A so that the base nodes come first
    A = baseFirst(A,base)
    bSize = len(base)


    #Find connected components and compress graph
    smA,comp = compressGraph(A,bSize)

    #Remove all sink components
    A,comp = delSink(A,bSize,comp=comp)

    #OUTDEGREE METHOD
    S = A.copy()
    moreLinks = True
    while moreLinks:
        moreLinks = False
        #Check the out-degree of each component
        for key in comp.keys():
            if comp[key][0] >= bSize:
                outDeg = S[:,comp[key]].sum() - S[comp[key]][:,comp[key]].sum()
                #If the out-degree is greater than 1, specialize the component
                if outDeg > 1:
                    moreLinks = True
                    n = S.shape[0]
                    otherNodes = list(set(range(n)).difference(set(comp[key])))
                    #Find all outgoing links
                    links = findLinks(S,comp[key],otherNodes)
                    for i in range(1,len(links)):
                        n = S.shape[0]
                        S = outSplzLink(S,bSize,links[i],links[i][1],comp[key])
                        #Add the new component to the component dictionary
                        comp[max(comp.keys())+1] = np.arange(n,n+len(comp[key]))
                        #Recompute out-degree if necessary
    return S

def inSpecialize(A,base):
    return outSpecialize(A.T,base).T

def findLinks(A,tail,tip):
    n = A.shape[0]

    rows = tip*len(tail)
    cols = []
    for c in tail:
        cols += [c]*len(tip)

    B = np.zeros_like(A)
    B[rows,cols] = A[rows,cols]

    #Find links
    linkRows,linkCols = np.where(B==1)
    links = zip(linkRows,linkCols)
    return links

def outSplzLink(A,bSize,edge,node,comp):

    #If the node being copied is in the base
    #return A unchanged
    if edge[1] < bSize:
        return A

    #If there is only one outgoing link from the component
    #return A unchanged
    if A[:,comp].sum() - A[comp][:,comp].sum() == 1:
        return A

    #Find sizes and locations
    whichNode = np.where(comp==node)[0]
    CLen = len(comp)
    n = A.shape[0]

    #Add the new component to the graph
    sA = block_diag(A,A[comp][:,comp])

    #Create new edge running from the copied component
    #to the graph
    newEdge = edge[0],n+whichNode

    sA[newEdge] = 1
    #Delete the old edge
    sA[edge] = 0


    #Keep all incoming edges
    keptLinks = A[comp,:]
    #Remove old intercomponent edges
    keptLinks[:,comp] = np.zeros((CLen,CLen))
    sA[-CLen:,:-CLen] = keptLinks

    return sA


"""
OTHER USEFUL FUNCTIONS
"""


def spectralRad(A):
    """
    Returns the spectral radius of matrix A
    """
    eigs = np.linalg.eig(A)[0]
    eigs = (eigs*eigs.conj())**.5
    return max(eigs)

def drawGraph(A):
    """Draws graph represented by adjacency matrix A"""
    m,n = A.shape
    labels = {}
    for i in range(n):
        labels[i]=str(i)
    gr = nx.from_numpy_matrix(A.T,create_using=nx.DiGraph())
    nx.draw(gr,arrows=True,node_color='#15b01a',labels=labels)
    plt.show()

def laplacian(A,normalize=False,randomWalk=False):
    """Returns the laplacian matrix of the graph induced by A"""
    degr = A.sum(axis=1)*1.

    if randomWalk:
        degr[degr!=0] = 1./degr[degr!=0]
        Dinv = np.diag(degr)
        return np.eye(A.shape[0]) - np.dot(Dinv,A)
    if normalize:
        degr = A.sum(axis=1)*1.
        degr[degr!=0] = 1./degr[degr!=0]
        Dinv = np.diag(degr)**.5
        return np.eye(A.shape[0]) - np.dot(Dinv,A).dot(Dinv)

    return np.diag(degr) - A

def randomGraph(n,base=True,bSize=None,stronglyCon=False):
    """
    Random Graph on n vertices with an optional
    random base set of vertices
    """
    A = (np.random.rand(n,n)>np.random.rand())*1.
    for j in range(n): A[j,j] = 0
    nodes = list(range(n))

    if stronglyCon:
        while not nx.is_strongly_connected(nx.DiGraph(A)):
            A = (np.random.rand(n,n)>np.random.rand())*1.
            for j in range(n): A[j,j] = 0
            nodes = list(range(n))

    if base:
        if bSize is None:
            bSize = np.random.randint(1,high=n)
        base = list(np.random.choice(nodes,replace=False,size=bSize))
        return A,base
    return A

def fiedler(A):
    """
    Returns the feedler eigenvalue
    """
    L = laplacian(A,randomWalk=0)
    eigs = np.linalg.eigvals(L)
    ind = np.argsort(np.abs(eigs))[1]
    fEig = eigs[ind]
    return fEig

def stableSpeci(n,grow=True):
    """
    Returns a random nxn adjacency matrix whose
    laplacian specializes stably
    """
    unstable=True
    while unstable:
        G,base = randomGraph(n)
        sG = specializeGraph(G,base)
        rhoG = spectralRad(laplacian(G))
        rhoSG = spectralRad(laplacian(sG))
        if sG.shape[0] > n:
            if np.isclose(rhoG,rhoSG):
                unstable=False

    return G,base

def eigCent(A):
    """
    Returns the centrality vector
    """
    lam,V = np.linalg.eig(A)
    v = V[:,np.argmax(lam)]
    v = v*(1./v[0])
    return v
