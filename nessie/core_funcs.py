"""
Module which contains the core functionality for the Nessie group finder.
"""

import networkx as nx
import numpy as np
from nessie_py import fof_links_fast


def _group_graph(links: dict):
    """
    Build a group cataog from a dictionary of links. Involves building the graph and finding
    connected subgraphs.
    """
    graph = nx.Graph()
    edges = zip(links["i"], links["j"])
    graph.add_edges_from(edges)

    components = nx.connected_components(graph)

    group_ids = []
    galaxy_ids = []
    for group_id, component in enumerate(components):
        for galaxy_id in component:
            group_ids.append(group_id)
            galaxy_ids.append(galaxy_id)

    # Mathing the 1-index from R
    group_table = {
        "galaxy_id": np.array(galaxy_ids),
        "group_id": np.array(group_ids) + 1,
    }
    return group_table


def _find_groups(
    ra: np.ndarray[float],
    dec: np.ndarray[float],
    comoving_distance: np.ndarray[float],
    linking_length_pos: np.ndarray[float],
    linking_length_los: np.ndarray[float],
) -> dict:
    """
    Identify groups within the given ra, dec, and comoving distance arrays, given the los and pos
    linking lengths.
    """
    links = fof_links_fast(
        ra, dec, comoving_distance, linking_length_pos, linking_length_los
    )
    return _group_graph(links)
