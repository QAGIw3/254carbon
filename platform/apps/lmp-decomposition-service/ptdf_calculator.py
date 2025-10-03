"""
PTDF (Power Transfer Distribution Factor) calculations.
"""
import logging
from typing import Dict, Any
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


class PTDFCalculator:
    """Calculate PTDF using DC power flow."""
    
    async def get_network_topology(self, iso: str) -> Dict[str, Any]:
        """
        Get network topology (nodes, lines, reactances).
        
        In production, would load from network model database.
        """
        # Mock network for demonstration
        # Real implementation would query actual network topology
        
        if iso == "PJM":
            return self._get_pjm_mock_network()
        elif iso == "MISO":
            return self._get_miso_mock_network()
        else:
            return self._get_generic_network()
    
    def _get_pjm_mock_network(self) -> Dict[str, Any]:
        """Mock PJM network topology."""
        return {
            "nodes": ["PJM.BUS1", "PJM.BUS2", "PJM.BUS3", "PJM.HUB.WEST"],
            "lines": [
                {"from": "PJM.BUS1", "to": "PJM.BUS2", "reactance": 0.1, "limit": 500},
                {"from": "PJM.BUS2", "to": "PJM.BUS3", "reactance": 0.15, "limit": 400},
                {"from": "PJM.BUS1", "to": "PJM.HUB.WEST", "reactance": 0.08, "limit": 600},
            ],
            "reference_bus": "PJM.HUB.WEST",
        }
    
    def _get_miso_mock_network(self) -> Dict[str, Any]:
        """Mock MISO network topology."""
        return {
            "nodes": ["MISO.BUS1", "MISO.BUS2", "MISO.HUB.INDIANA"],
            "lines": [
                {"from": "MISO.BUS1", "to": "MISO.BUS2", "reactance": 0.12, "limit": 450},
                {"from": "MISO.BUS2", "to": "MISO.HUB.INDIANA", "reactance": 0.10, "limit": 500},
            ],
            "reference_bus": "MISO.HUB.INDIANA",
        }
    
    def _get_generic_network(self) -> Dict[str, Any]:
        """Generic network topology."""
        return {
            "nodes": ["BUS1", "BUS2", "BUS3"],
            "lines": [
                {"from": "BUS1", "to": "BUS2", "reactance": 0.1, "limit": 500},
                {"from": "BUS2", "to": "BUS3", "reactance": 0.1, "limit": 500},
            ],
            "reference_bus": "BUS1",
        }
    
    def calculate_ptdf(
        self,
        source_node: str,
        sink_node: str,
        constraint_id: str,
        network: Dict[str, Any],
    ) -> float:
        """
        Calculate PTDF using DC power flow approximation.
        
        PTDF = (change in line flow) / (1 MW injection)
        
        Using DC power flow:
        - Power flow on line l: P_l = (θ_i - θ_j) / X_ij
        - PTDF_l = ∂P_l / ∂P_injection
        """
        # Build network graph
        G = nx.Graph()
        
        for line in network["lines"]:
            G.add_edge(
                line["from"],
                line["to"],
                reactance=line["reactance"],
                limit=line["limit"],
            )
        
        # Build admittance matrix (B matrix)
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        B = np.zeros((n_nodes, n_nodes))
        
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j and G.has_edge(node_i, node_j):
                    reactance = G[node_i][node_j]["reactance"]
                    susceptance = 1.0 / reactance
                    B[i, j] = -susceptance
                    B[i, i] += susceptance
        
        # Remove reference bus row/column
        ref_bus = network["reference_bus"]
        if ref_bus in nodes:
            ref_idx = nodes.index(ref_bus)
            B_reduced = np.delete(np.delete(B, ref_idx, axis=0), ref_idx, axis=1)
            nodes_reduced = [n for i, n in enumerate(nodes) if i != ref_idx]
        else:
            B_reduced = B
            nodes_reduced = nodes
        
        # Calculate PTDF
        try:
            # Sensitivity matrix: θ = B^(-1) * P_injection
            B_inv = np.linalg.inv(B_reduced)
            
            # Find source and sink indices
            if source_node in nodes_reduced:
                source_idx = nodes_reduced.index(source_node)
            else:
                source_idx = 0  # Reference bus
            
            if sink_node in nodes_reduced:
                sink_idx = nodes_reduced.index(sink_node)
            else:
                sink_idx = 0  # Reference bus
            
            # PTDF for constraint (simplified: use first line as constraint)
            # In production, would match constraint_id to actual line
            
            # For demonstration, calculate sensitivity to first line
            if network["lines"]:
                line = network["lines"][0]
                from_node = line["from"]
                to_node = line["to"]
                reactance = line["reactance"]
                
                # Find indices
                if from_node in nodes_reduced:
                    from_idx = nodes_reduced.index(from_node)
                    to_idx = nodes_reduced.index(to_node) if to_node in nodes_reduced else -1
                    
                    if to_idx >= 0:
                        # PTDF = (θ_from - θ_to) / X for 1 MW injection
                        ptdf = (B_inv[from_idx, source_idx] - B_inv[to_idx, source_idx]) / reactance
                        return float(ptdf)
            
            # Fallback
            return 0.5  # Mock PTDF value
            
        except np.linalg.LinAlgError:
            logger.error("Singular matrix - network topology issue")
            return 0.0

