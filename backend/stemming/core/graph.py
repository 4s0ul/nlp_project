from datetime import datetime, timezone

import numpy as np
import networkx as nx
from typing import List, Optional
from loguru import logger
from sqlalchemy import delete
from sqlmodel import Session, select
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist, pdist
from matplotlib import pyplot as plt
import hdbscan
import umap
from stemming.core.models import Word, Description, Embedding, Triplet, Lang, get_session, DEFAULT_VECTOR_DIM, GraphEdge, \
    GraphNode, generate_sha256_id
from stemming.core.vectorization import nlp_ru_md


class Data4Graph:
    def __init__(self, word_id: str, term: str, term_vectorized: np.ndarray, definition_vector: np.ndarray):
        self.word_id = word_id
        self.term = term
        self.term_vectorized = term_vectorized
        self.definition_vector = definition_vector

    @classmethod
    def create_from_db(cls, word: Word, description: Optional[Description], embedding: Optional[Embedding]):
        term_vector = np.zeros(DEFAULT_VECTOR_DIM)  # Fallback
        if word.language == Lang.Russian.value:
            doc = nlp_ru_md(word.lemmatized_text)
            term_vector = doc.vector if doc.has_vector and doc.vector_norm > 0 else term_vector
        definition_vector = np.array(embedding.embedding) if embedding else np.zeros(DEFAULT_VECTOR_DIM)
        return cls(
            word_id=word.id,
            term=word.raw_text,
            term_vectorized=term_vector,
            definition_vector=definition_vector
        )


class GraphBuilder:
    CONFIG = {
        'max_semantic_relations': 35,
        'semantic_threshold': 0.93,
        'top_nodes_to_visualize': 100,
        'min_cluster_size': 3
    }

    def __init__(self, language: Lang, session: Session):
        self.language = language.value
        self.graph = nx.DiGraph()
        self.node_id_map = {}  # Maps term to graph_node.id
        self.session = session

    def _pad_vectors(self, vectors: List[np.ndarray]) -> np.ndarray:
        max_len = max(vec.shape[0] for vec in vectors)
        return np.array([
            np.pad(vec, (0, max_len - vec.shape[0]), mode='constant') if vec.shape[0] < max_len else vec[:max_len]
            for vec in vectors
        ])

    def save_graph_to_db(self):
        """Saves the graph's nodes and edges to graph_nodes and graph_edges tables."""
        # Clear existing graph data for this language
        self.session.exec(
            delete(GraphEdge).where(GraphEdge.language == self.language)
        )
        self.session.exec(
            delete(GraphNode).where(GraphNode.language == self.language)
        )
        self.session.commit()

        # Save nodes
        for term, data in self.graph.nodes(data=True):
            node_id = generate_sha256_id(f"{term}_{self.language}")
            graph_node = GraphNode(
                id=node_id,
                word_id=data["word_id"],
                term=term,
                language=self.language,
                centroid_weight=float(data["centroid_weight"]),
                centrality_weight=float(data["centrality_weight"]),
                combined_weight=float(data["combined_weight"]),
                created_at=datetime.now(timezone.utc)
            )
            self.session.add(graph_node)
            self.node_id_map[term] = node_id

        # Save edges
        for u, v, data in self.graph.edges(data=True):
            edge_id = generate_sha256_id(f"{u}_{v}_{self.language}_{data['type']}")
            graph_edge = GraphEdge(
                id=edge_id,
                source_node_id=self.node_id_map[u],
                target_node_id=self.node_id_map[v],
                type=data["type"],
                predicate=data.get("predicate"),
                weight=float(data["weight"]),
                language=self.language,
                triplet_id=data.get("triplet_id"),
                created_at=datetime.now(timezone.utc)
            )
            self.session.add(graph_edge)

        self.session.commit()
        logger.info(f"Saved graph to DB: {len(self.node_id_map)} nodes, {self.graph.number_of_edges()} edges")

    def build_graph(self):
        words = self.session.exec(
            select(Word).where(Word.language == self.language)
        ).all()
        terms = []
        for word in words:
            description = self.session.exec(
                select(Description).where(Description.word_id == word.id)
            ).first()
            embedding = self.session.exec(
                select(Embedding).where(Embedding.description_id == description.id)
            ).first() if description else None
            terms.append(Data4Graph.create_from_db(word, description, embedding))

        self.terms = terms
        term_vectors = normalize(self._pad_vectors([x.term_vectorized for x in self.terms]))
        def_vectors = normalize(self._pad_vectors([x.definition_vector for x in self.terms]))
        physical_terms = [x.term for x in self.terms]
        word_ids = [x.word_id for x in self.terms]

        X_combined = np.hstack([term_vectors * 0.7, def_vectors * 0.3])
        X_combined = normalize(X_combined)

        reducer = umap.UMAP(n_components=8, metric='cosine', random_state=42)
        X_reduced = reducer.fit_transform(X_combined)

        n_points = len(X_combined)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(self.CONFIG['min_cluster_size'], int(n_points * 0.015)),
            min_samples=max(2, int(n_points * 0.005)),
            metric='euclidean',
            cluster_selection_method='eom'
        )
        semantic_clusters = clusterer.fit_predict(X_reduced)

        valid_mask = semantic_clusters != -1
        valid_clusters = np.unique(semantic_clusters[valid_mask])
        n_valid_clusters = len(valid_clusters)
        if valid_mask.sum() > 1 and n_valid_clusters > 1:
            score = silhouette_score(X_reduced[valid_mask], semantic_clusters[valid_mask], metric='euclidean')
            logger.info(
                f"HDBSCAN Silhouette Score: {score:.3f}, Clusters: {n_valid_clusters}, Noise Points: {(semantic_clusters == -1).sum()}"
            )
        else:
            logger.warning("HDBSCAN failed. Falling back to KMeans.")
            best_score, best_k, best_clusters = -1, 2, None
            for k in range(2, min(20, n_points // 10)):
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(X_reduced)
                score = silhouette_score(X_reduced, clusters, metric='euclidean')
                if score > best_score:
                    best_score, best_k, best_clusters = score, k, clusters
            semantic_clusters = best_clusters
            valid_clusters = np.unique(semantic_clusters)
            logger.info(f"KMeans Silhouette Score: {best_score:.3f}, Clusters: {best_k}")

        semantic_centroids = np.array([
            X_reduced[semantic_clusters == c].mean(axis=0) for c in valid_clusters
        ]) if len(valid_clusters) > 0 else np.array([])

        node_weights = {}
        for i, (term, word_id) in enumerate(zip(physical_terms, word_ids)):
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
                'word_id': word_id,
                'centroid_weight': centroid_weight,
                'centrality_weight': centrality,
                'combined_weight': 0.7 * centroid_weight + 0.3 * centrality
            }
            self.graph.add_node(term, **node_weights[term])

        triplets = self.session.exec(
            select(Triplet).where(Triplet.language == self.language).where(Triplet.subject_type == "word")
        ).all()
        term_to_id = {term: word_id for term, word_id in zip(physical_terms, word_ids)}
        for triplet in triplets:
            subject_term = next(
                (term for term, wid in term_to_id.items() if wid == triplet.subject_id), None
            )
            if not subject_term or subject_term not in self.graph:
                continue
            if triplet.object_id:
                object_term = next(
                    (term for term, wid in term_to_id.items() if wid == triplet.object_id), None
                )
                if not object_term or object_term not in self.graph:
                    continue
            else:
                object_term = triplet.object_literal_raw or triplet.object_literal_lemma
                if not object_term or object_term not in self.graph:
                    continue
            predicate = triplet.predicate_lemma or triplet.predicate_raw
            self.graph.add_edge(
                subject_term,
                object_term,
                type='triplet',
                predicate=predicate,
                weight=1.0,
                triplet_id=triplet.id
            )

        term_similarities = 1 - cdist(term_vectors, term_vectors, metric='cosine')
        def_similarities = 1 - cdist(def_vectors, def_vectors, metric='cosine')

        distance_matrix = pdist(X_combined, metric='euclidean')
        Z = linkage(distance_matrix, method='ward')
        threshold = np.percentile(Z[:, 2], 90)
        hierarchical_clusters = fcluster(Z, t=threshold, criterion='distance')

        combined_distances = 0.5 * cdist(term_vectors, term_vectors, metric='cosine') + 0.5 * cdist(def_vectors,
                                                                                                    def_vectors,
                                                                                                    metric='cosine')

        for i, term1 in enumerate(physical_terms):
            for j, term2 in enumerate(physical_terms):
                if i < j and hierarchical_clusters[i] == hierarchical_clusters[j]:
                    if term1 not in node_weights or term2 not in node_weights:
                        continue
                    semantic_sim = 1 - combined_distances[i, j]
                    structural_sim = (1 - (
                        Z[np.where((Z[:, 0] == i) & (Z[:, 1] == j))[0][0], 2]
                        if np.any((Z[:, 0] == i) & (Z[:, 1] == j))
                        else threshold
                    ) / threshold)
                    weight = (
                            0.4 * semantic_sim +
                            0.4 * structural_sim +
                            0.2 * (node_weights[term1]['combined_weight'] + node_weights[term2][
                        'combined_weight']) / 2
                    )
                    if weight > self.CONFIG['semantic_threshold']:
                        self.graph.add_edge(
                            term1,
                            term2,
                            type='hierarchical',
                            weight=weight,
                            distance=combined_distances[i, j],
                            merge_level=structural_sim
                        )

        term_threshold = np.percentile(term_similarities, 95)
        def_threshold = np.percentile(def_similarities, 95)
        min_edge_weight = 0.7
        cluster_to_indices = {c: np.where(semantic_clusters == c)[0] for c in valid_clusters}
        for cluster_idx, indices in cluster_to_indices.items():
            edges = [
                (
                    i, j,
                    term_similarities[i, j],
                    def_similarities[i, j],
                    1 / (1 + np.linalg.norm(
                        semantic_centroids[np.where(valid_clusters == cluster_idx)[0]] - (
                                    X_reduced[i] + X_reduced[j]) / 2
                    ))
                )
                for idx_i, i in enumerate(indices)
                for idx_j, j in enumerate(indices[:idx_i])
            ]
            for i, j, ts, ds, cw in edges:
                if ts > term_threshold or ds > def_threshold:
                    weight = 0.5 * ts + 0.3 * ds + 0.2 * cw
                    if weight > min_edge_weight:
                        self.graph.add_edge(
                            physical_terms[i],
                            physical_terms[j],
                            type='semantic',
                            weight=weight,
                            term_similarity=ts,
                            def_similarity=ds,
                            cluster_weight=cw
                        )

        for node in list(self.graph.nodes):
            edges = list(self.graph.edges(node, data=True))
            if len(edges) > 10:
                edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)[:10]
                self.graph.remove_edges_from(
                    [(u, v) for u, v, d in self.graph.edges(node, data=True) if (u, v, d) not in edges]
                )
        self.graph.remove_edges_from(
            [(u, v) for u, v, d in self.graph.edges(data=True) if d['weight'] < min_edge_weight]
        )

        logger.info(
            f"Graph [Lang.{self.language}]: Nodes={self.graph.number_of_nodes()}, Edges={self.graph.number_of_edges()}"
        )

        # Save graph to database
        self.save_graph_to_db()

        return self.graph

    def visualize_graph(self, output_file: str = 'knowledge_graph.png'):
        logger.info(
            f"Visualizing graph [Lang.{self.language}]: Nodes={self.graph.number_of_nodes()}, Edges={self.graph.number_of_edges()}"
        )
        top_nodes = sorted(self.graph.degree(weight='weight'), key=lambda x: x[1], reverse=True)[
                    :self.CONFIG['top_nodes_to_visualize']
                    ]
        subgraph = self.graph.subgraph([n for n, _ in top_nodes])

        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(subgraph, k=0.15, iterations=20)

        nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color='lightblue')
        edge_types = set(d['type'] for u, v, d in subgraph.edges(data=True))
        for edge_type in edge_types:
            edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d['type'] == edge_type]
            labels = {
                (u, v): d['predicate'] if d['type'] == 'triplet' else ''
                for u, v, d in subgraph.edges(data=True) if (u, v) in edges
            }
            nx.draw_networkx_edges(
                subgraph,
                pos,
                edgelist=edges,
                width=[d['weight'] * 2 for u, v, d in subgraph.edges(data=True) if (u, v) in edges],
                edge_color='blue' if edge_type == 'triplet' else 'gray',
                style='solid' if edge_type == 'triplet' else 'dashed',
                alpha=0.7
            )
            if edge_type == 'triplet':
                nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=labels, font_size=6)

        nx.draw_networkx_labels(subgraph, pos, font_size=8)
        plt.title(f"Knowledge Graph: Top-{self.CONFIG['top_nodes_to_visualize']} Terms ({self.language})")
        plt.savefig(output_file)
        plt.show()

def build_knowledge_graph(language: Lang = Lang.Russian) -> nx.DiGraph:
    session = next(get_session())
    try:
        builder = GraphBuilder(language, session)
        return builder.build_graph()
    finally:
        session.close()

def visualize_knowledge_graph(language: Lang = Lang.Russian, output_file: str = 'knowledge_graph.png'):
    session = next(get_session())
    try:
        builder = GraphBuilder(language, session)
        graph = builder.build_graph()
        builder.visualize_graph(output_file)
    finally:
        session.close()