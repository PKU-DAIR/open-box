#    Copyright (C) 2004-2010 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#
# Add platform dependent shared library path to sys.path
#
# Modified by Matthias Feurer for the package HPOlibConfigSpace

from __future__ import absolute_import

# Release data
from ConfigSpace.nx.release import authors, license, date, version

__author__ = '%s <%s>\n%s <%s>\n%s <%s>' % \
              (authors['Hagberg'] + authors['Schult'] + authors['Swart'])
__license__ = license

__date__ = date
__version__ = version

from ConfigSpace.nx.exception import (
    NetworkXException, NetworkXError,
    NetworkXPointlessConcept, NetworkXAlgorithmError,
    NetworkXUnfeasible, NetworkXNoPath,
    NetworkXUnbounded, NetworkXNotImplemented
)

#  import ConfigSpace.nx.classes
from ConfigSpace.nx.classes import (
    Graph, DiGraph
)

from ConfigSpace.nx.algorithms import (
    descendants, ancestors, topological_sort, topological_sort_recursive,
    is_directed_acyclic_graph, is_aperiodic, simple_cycles,
    strongly_connected_components
)

__all__ = [
    "NetworkXException",
    "NetworkXError",
    "NetworkXPointlessConcept",
    "NetworkXAlgorithmError",
    "NetworkXUnfeasible",
    "NetworkXNoPath",
    "NetworkXUnbounded",
    "NetworkXNotImplemented",
    "Graph",
    "DiGraph",
    "descendants",
    "ancestors",
    "topological_sort",
    "topological_sort_recursive",
    "is_directed_acyclic_graph",
    "is_aperiodic",
    "simple_cycles",
    "strongly_connected_components"
]
