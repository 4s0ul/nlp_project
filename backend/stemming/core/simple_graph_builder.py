from stemming.core.models import Data4Graph, Lang, TermVectorized

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist, pdist

from matplotlib import pyplot as plt
from loguru import logger
from typing import List
from tqdm import tqdm
import numpy as np

import networkx as nx
import hdbscan
import umap


class GraphBuilder:
    terms: List

    CONFIG = {
        'max_semantic_relations': 35,
        'semantic_threshold': 0.93,
        'top_nodes_to_visualize': 100,
        'min_cluster_size': 3
    }

    def __init__(self, language: Lang):
        self.language = language.value
        self.graph = nx.DiGraph()

    def _pad_vectors(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        max_len = max(vec.shape[0] for vec in vectors)
        return [
            np.pad(vec, (0, max_len - vec.shape[0]), mode='constant') if vec.shape[0] < max_len else vec[:max_len]
            for vec in vectors]

    def build_graph(self, terms_vectorized: List[TermVectorized]):
        # Extract and normalize vectors

        self.terms = list(map(Data4Graph.create_from_term_vectorized, terms_vectorized))
        term_vectors = normalize(np.array(self._pad_vectors([np.array(x.term_vectorized) for x in self.terms])))
        def_vectors = normalize(np.array(self._pad_vectors([np.array(x.definition_vector) for x in self.terms])))
        physical_terms = [x.term for x in self.terms]

        # Combine vectors with weighted concatenation
        X_combined = np.hstack([term_vectors * 0.7, def_vectors * 0.3])
        X_combined = normalize(X_combined)

        # Dimensionality reduction with UMAP
        reducer = umap.UMAP(n_components=8, metric='cosine', random_state=42)
        X_reduced = reducer.fit_transform(X_combined)

        # HDBSCAN clustering with dynamic parameters
        n_points = len(X_combined)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, int(n_points * 0.015)),  # ~1.5% of points
            min_samples=max(3, int(n_points * 0.005)),  # ~0.5% of points
            metric='euclidean',
            cluster_selection_method='eom'
        )
        semantic_clusters = clusterer.fit_predict(X_reduced)

        # Evaluate HDBSCAN
        valid_mask = semantic_clusters != -1
        valid_clusters = np.unique(semantic_clusters[valid_mask])
        n_valid_clusters = len(valid_clusters)
        if valid_mask.sum() > 1 and n_valid_clusters > 1:
            score = silhouette_score(X_reduced[valid_mask], semantic_clusters[valid_mask], metric='euclidean')
            print(
                f"HDBSCAN Silhouette Score: {score:.3f}, Clusters: {n_valid_clusters}, Noise Points: {(semantic_clusters == -1).sum()}")
        else:
            print("HDBSCAN failed to produce valid clusters. Falling back to KMeans.")
            # Fallback to KMeans
            best_score, best_k, best_clusters = -1, 2, None
            for k in range(2, min(20, n_points // 10)):
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(X_reduced)
                score = silhouette_score(X_reduced, clusters, metric='euclidean')
                if score > best_score:
                    best_score, best_k, best_clusters = score, k, clusters
            semantic_clusters = best_clusters
            valid_clusters = np.unique(semantic_clusters)
            print(f"KMeans Silhouette Score: {best_score:.3f}, Clusters: {best_k}")

        # Compute centroids for valid clusters
        semantic_centroids = np.array([X_reduced[semantic_clusters == c].mean(axis=0) for c in valid_clusters]) if len(
            valid_clusters) > 0 else np.array([])

        # Add nodes with weights
        node_weights = {}
        for i, term in enumerate(physical_terms):
            cluster_id = semantic_clusters[i]

            if cluster_id == -1:
                continue

            centroid_idx = np.where(valid_clusters == cluster_id)[0][0]
            centroid_dist = np.linalg.norm(X_reduced[i] - semantic_centroids[centroid_idx])
            same_cluster = X_reduced[semantic_clusters == cluster_id]
            intra_dists = cdist([X_reduced[i]], same_cluster, metric='cosine')[0]
            centrality = 1 / (1 + np.mean(intra_dists))
            centroid_weight = 1 / (1 + centroid_dist)
            node_weights[term] = {
                'centroid_weight': centroid_weight,
                'centrality_weight': centrality,
                'combined_weight': 0.7 * centroid_weight + 0.3 * centrality
            }
            self.graph.add_node(term, **node_weights[term])

        # Compute similarities
        term_similarities = 1 - cdist(term_vectors, term_vectors, metric='cosine')
        def_similarities = 1 - cdist(def_vectors, def_vectors, metric='cosine')

        # Compute condensed distance matrix
        distance_matrix = pdist(X_combined, metric='euclidean')

        # Perform hierarchical clustering
        Z = linkage(distance_matrix, method='ward')

        # Automatic threshold selection (90th percentile of distances)
        threshold = np.percentile(Z[:, 2], 90)
        hierarchical_clusters = fcluster(Z, t=threshold, criterion='distance')

        # Distance matrices
        term_distances = cdist(term_vectors, term_vectors, metric='cosine')  # Changed to cosine for text
        def_distances = cdist(def_vectors, def_vectors, metric='cosine')
        combined_distances = 0.5 * term_distances + 0.5 * def_distances

        # HIERARCHICAL EDGES (using dendrogram-based clusters)
        logger.info("Building hierarchical edges...")
        for i, term1 in enumerate(physical_terms):
            for j, term2 in enumerate(physical_terms):
                if i < j and hierarchical_clusters[i] == hierarchical_clusters[j]:
                    if node_weights.get(term1) is None or \
                            node_weights.get(term2) is None:
                        continue
                    # Combine semantic and structural similarity
                    semantic_sim = 1 - combined_distances[i, j]
                    structural_sim = (1 - (Z[np.where((Z[:, 0] == i) & (Z[:, 1] == j))[0][0], 2]
                                           if np.any((Z[:, 0] == i) & (Z[:, 1] == j))
                                           else threshold) / threshold)

                    weight = (0.4 * semantic_sim +
                              0.4 * structural_sim +
                              0.2 * (node_weights[term1]['combined_weight'] +
                                     node_weights[term2]['combined_weight']) / 2)

                    self.graph.add_edge(term1, term2,
                                        type='hierarchical',
                                        weight=weight,
                                        distance=combined_distances[i, j],
                                        merge_level=structural_sim)

        # Stricter thresholds (top 5%)
        term_threshold = np.percentile(term_similarities, 95)
        def_threshold = np.percentile(def_similarities, 95)
        min_edge_weight = 0.7

        # Build semantic edges within clusters
        cluster_to_indices = {c: np.where(semantic_clusters == c)[0] for c in valid_clusters}
        for cluster_idx, indices in tqdm(cluster_to_indices.items(), desc="Building semantic edges"):
            edges = [
                (i, j, term_similarities[i, j], def_similarities[i, j],
                 1 / (1 + np.linalg.norm(semantic_centroids[np.where(valid_clusters == cluster_idx)[0]] - (
                         X_reduced[i] + X_reduced[j]) / 2)))
                for idx_i, i in enumerate(indices)
                for idx_j, j in enumerate(indices[:idx_i])
            ]
            for i, j, ts, ds, cw in edges:
                if ts > term_threshold or ds > def_threshold:
                    weight = 0.5 * ts + 0.3 * ds + 0.2 * cw
                    if weight > min_edge_weight:
                        self.graph.add_edge(
                            physical_terms[i], physical_terms[j],
                            type='semantic',
                            weight=weight,
                            term_similarity=ts,
                            def_similarity=ds,
                            cluster_weight=cw
                        )

        # Cross-cluster connections
        if len(valid_clusters) > 1:
            cluster_similarities = 1 - cdist(semantic_centroids, semantic_centroids, metric='cosine')
            top_pairs = np.argsort(cluster_similarities.ravel())[::-1][:self.CONFIG['max_semantic_relations']]
            for idx in top_pairs:
                c1, c2 = divmod(idx, len(valid_clusters))
                if c1 < c2:
                    c1_indices = cluster_to_indices[valid_clusters[c1]][:3]
                    c2_indices = cluster_to_indices[valid_clusters[c2]][:3]
                    for i in c1_indices:
                        for j in c2_indices:
                            ts, ds = term_similarities[i, j], def_similarities[i, j]
                            weight = 0.5 * ts + 0.3 * ds + 0.2 * cluster_similarities[c1, c2]
                            if weight > min_edge_weight:
                                self.graph.add_edge(
                                    physical_terms[i], physical_terms[j],
                                    type='cross_cluster',
                                    weight=weight,
                                    term_similarity=ts,
                                    def_similarity=ds
                                )

        # Prune edges
        for node in list(self.graph.nodes):
            edges = list(self.graph.edges(node, data=True))
            if len(edges) > 10:
                edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)[:10]
                self.graph.remove_edges_from(
                    [(u, v) for u, v, d in self.graph.edges(node, data=True) if (u, v, d) not in edges])
        self.graph.remove_edges_from(
            [(u, v) for u, v, d in self.graph.edges(data=True) if d['weight'] < min_edge_weight])

        # Diagnostics
        print(f"Graph [Lang.Russian]: Nodes={self.graph.number_of_nodes()}, Edges={self.graph.number_of_edges()}")

        return self.graph

    def visualize_graph(self):
        print(
            f"Graph [{self.language}]: Nodes={self.graph.number_of_nodes()}, Edges={self.graph.number_of_edges()}")

        top_nodes = sorted(self.graph.degree(weight='weight'), key=lambda x: x[1], reverse=True)[
                    :self.CONFIG['top_nodes_to_visualize']]
        subgraph = self.graph.subgraph([n for n, _ in top_nodes])

        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_size=300,
                font_size=8, width=[d['weight'] * 2 for u, v, d in subgraph.edges(data=True)])
        plt.title(f"Top-{self.CONFIG['top_nodes_to_visualize']} terms ({self.language})")
        plt.savefig('graph.png')
        plt.show()
