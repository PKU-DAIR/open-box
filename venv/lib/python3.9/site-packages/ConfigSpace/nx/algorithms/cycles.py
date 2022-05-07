"""
========================
Cycle finding algorithms
========================
"""
#    Copyright (C) 2010-2012 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
from collections import defaultdict
import ConfigSpace.nx

__all__ = ['simple_cycles']
__author__ = "\n".join(['Jon Olav Vik <jonovik@gmail.com>',
                        'Dan Schult <dschult@colgate.edu>',
                        'Aric Hagberg <hagberg@lanl.gov>'])


def simple_cycles(G):
    """Find simple cycles (elementary circuits) of a directed graph.

    An simple cycle, or elementary circuit, is a closed path where no
    node appears twice, except that the first and last node are the same.
    Two elementary circuits are distinct if they are not cyclic permutations
    of each other.

    This is a nonrecursive, iterator/generator version of Johnson's
    algorithm [1]_.  There may be better algorithms for some cases [2]_ [3]_.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    Returns
    -------
    cycle_generator: generator
       A generator that produces elementary cycles of the graph.  Each cycle is
       a list of nodes with the first and last nodes being the same.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)])
    >>> list(nx.simple_cycles(G))
    [[2], [2, 1], [2, 0], [2, 0, 1], [0]]

    Notes
    -----
    The implementation follows pp. 79-80 in [1]_.

    The time complexity is O((n+e)(c+1)) for n nodes, e edges and c
    elementary circuits.

    To filter the cycles so that they don't include certain nodes or edges,
    copy your graph and eliminate those nodes or edges before calling.
    >>> copyG = G.copy()
    >>> copyG.remove_nodes_from([1])
    >>> copyG.remove_edges_from([(0,1)])
    >>> list(nx.simple_cycles(copyG))
    [[2], [2, 0], [0]]

    References
    ----------
    .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       http://dx.doi.org/10.1137/0204007

    .. [2] Enumerating the cycles of a digraph: a new preprocessing strategy.
       G. Loizou and P. Thanish, Information Sciences, v. 27, 163-182, 1982.

    .. [3] A search strategy for the elementary cycles of a directed graph.
       J.L. Szwarcfiter and P.E. Lauer, BIT NUMERICAL MATHEMATICS,
       v. 16, no. 2, 192-204, 1976.

    See Also
    --------
    cycle_basis
    """
    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()

    # Johnson's algorithm requires some ordering of the nodes.
    # We assign the arbitrary ordering given by the strongly connected comps
    # There is no need to track the ordering as each node removed as processed.
    subG = G.copy()   # save the actual graph so we can mutate it here
    sccs = ConfigSpace.nx.strongly_connected_components(subG)
    while sccs:
        scc = sccs.pop()
        # order of scc determines ordering of nodes
        startnode = scc.pop()
        # Processing node runs "circuit" routine from recursive version
        path = [startnode]
        blocked = set()  # vertex: blocked from search?
        closed = set()  # nodes involved in a cycle
        blocked.add(startnode)
        B = defaultdict(set)  # graph portions that yield no elementary circuit
        stack = [(startnode, list(subG[startnode]))]  # subG gives component nbrs
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs:
                nextnode = nbrs.pop()
#                    print thisnode,nbrs,":",nextnode,blocked,B,path,stack,startnode
#                    f=raw_input("pause")
                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)
#                        print "Found a cycle",path,closed
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(subG[nextnode])))
                    blocked.add(nextnode)
                    continue
            # done with nextnode... look for more neighbors
            if not nbrs:  # no more nbrs
                if thisnode in closed:
                    _unblock(thisnode, blocked, B)
                else:
                    for nbr in G[thisnode]:
                        if thisnode not in B[nbr]:
                            B[nbr].add(thisnode)
                stack.pop()
#                assert path[-1]==thisnode
                path.pop()
        # done processing this node
        subG.remove_node(startnode)
        H = subG.subgraph(scc)  # make smaller to avoid work in SCC routine
        sccs.extend(ConfigSpace.nx.strongly_connected_components(H))
