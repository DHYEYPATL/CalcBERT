"""
Vector Store for v2 Backend - Semantic similarity search for expenses.
Primary: Pinecone cloud vector database
Fallback: Local sklearn-based vector store with disk persistence
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from backend.config import settings

# Try to import Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠ Pinecone not available, using local fallback vector store")

# Sklearn for fallback
from sklearn.neighbors import NearestNeighbors


class VectorStore:
    """
    Vector store for semantic similarity search.
    Supports Pinecone (primary) and local sklearn fallback.
    """
    
    def __init__(self):
        self.mode = None
        self.pinecone_index = None
        self.local_vectors = []
        self.local_metadata = []
        self.local_nn = None
        self._initialize()
    
    def _initialize(self):
        """Initialize vector store (Pinecone or fallback)."""
        # Try Pinecone first if configured
        if settings.USE_PINECONE and PINECONE_AVAILABLE:
            try:
                self._init_pinecone()
                self.mode = "pinecone"
                print(f"✓ Vector store initialized with Pinecone (index: {settings.PINECONE_INDEX_NAME})")
                return
            except Exception as e:
                print(f"⚠ Pinecone initialization failed: {e}")
                print("  Falling back to local vector store")
        
        # Fallback to local
        self._init_local()
        self.mode = "local"
        print(f"✓ Vector store initialized with local fallback")
    
    def _init_pinecone(self):
        """Initialize Pinecone vector database."""
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not configured")
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Check if index exists, create if not
        index_name = settings.PINECONE_INDEX_NAME
        
        if index_name not in pc.list_indexes().names():
            print(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=settings.VECTOR_DIMENSION,
                metric=settings.VECTOR_METRIC,
                spec=ServerlessSpec(
                    cloud='aws',
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
        
        self.pinecone_index = pc.Index(index_name)
    
    def _init_local(self):
        """Initialize local vector store with sklearn."""
        # Try to load existing store
        if os.path.exists(settings.VECTOR_FALLBACK_PATH):
            try:
                with open(settings.VECTOR_FALLBACK_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.local_vectors = data.get('vectors', [])
                    self.local_metadata = data.get('metadata', [])
                print(f"  Loaded {len(self.local_vectors)} vectors from disk")
            except Exception as e:
                print(f"  Could not load existing store: {e}")
                self.local_vectors = []
                self.local_metadata = []
        
        # Initialize NearestNeighbors if we have vectors
        if len(self.local_vectors) > 0:
            self.local_nn = NearestNeighbors(
                n_neighbors=min(5, len(self.local_vectors)),
                metric='cosine'
            )
            self.local_nn.fit(np.array(self.local_vectors))
    
    def upsert_embedding(
        self,
        key: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store an embedding vector with metadata.
        
        Args:
            key: Unique identifier for the vector
            vector: Embedding vector
            metadata: Associated metadata
        """
        if self.mode == "pinecone":
            self._pinecone_upsert(key, vector, metadata)
        else:
            self._local_upsert(key, vector, metadata)
    
    def _pinecone_upsert(
        self,
        key: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ):
        """Upsert to Pinecone."""
        self.pinecone_index.upsert(
            vectors=[(key, vector, metadata)]
        )
    
    def _local_upsert(
        self,
        key: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ):
        """Upsert to local store."""
        # Add key to metadata
        metadata['_key'] = key
        
        # Check if key exists, update if so
        existing_idx = None
        for i, meta in enumerate(self.local_metadata):
            if meta.get('_key') == key:
                existing_idx = i
                break
        
        if existing_idx is not None:
            # Update existing
            self.local_vectors[existing_idx] = vector
            self.local_metadata[existing_idx] = metadata
        else:
            # Add new
            self.local_vectors.append(vector)
            self.local_metadata.append(metadata)
        
        # Rebuild NearestNeighbors
        if len(self.local_vectors) > 0:
            self.local_nn = NearestNeighbors(
                n_neighbors=min(5, len(self.local_vectors)),
                metric='cosine'
            )
            self.local_nn.fit(np.array(self.local_vectors))
        
        # Persist to disk
        self._save_local()
    
    def query_similar(
        self,
        vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar vectors.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of similar items with metadata and scores
        """
        if self.mode == "pinecone":
            return self._pinecone_query(vector, top_k)
        else:
            return self._local_query(vector, top_k)
    
    def _pinecone_query(
        self,
        vector: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Query Pinecone."""
        results = self.pinecone_index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        
        similar = []
        for match in results.matches:
            similar.append({
                'id': match.id,
                'score': float(match.score),
                'metadata': match.metadata
            })
        
        return similar
    
    def _local_query(
        self,
        vector: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Query local store."""
        if len(self.local_vectors) == 0:
            return []
        
        # Adjust top_k if we have fewer vectors
        k = min(top_k, len(self.local_vectors))
        
        # Find nearest neighbors
        distances, indices = self.local_nn.kneighbors(
            [vector],
            n_neighbors=k
        )
        
        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            # Convert distance to similarity score (1 - cosine distance)
            score = 1.0 - dist
            similar.append({
                'id': self.local_metadata[idx].get('_key', f'item_{idx}'),
                'score': float(score),
                'metadata': {k: v for k, v in self.local_metadata[idx].items() if k != '_key'}
            })
        
        return similar
    
    def _save_local(self):
        """Save local store to disk."""
        try:
            os.makedirs(os.path.dirname(settings.VECTOR_FALLBACK_PATH), exist_ok=True)
            with open(settings.VECTOR_FALLBACK_PATH, 'wb') as f:
                pickle.dump({
                    'vectors': self.local_vectors,
                    'metadata': self.local_metadata
                }, f)
        except Exception as e:
            print(f"⚠ Could not save local vector store: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get vector store status.
        
        Returns:
            Status information
        """
        status = {
            'mode': self.mode,
            'pinecone_available': PINECONE_AVAILABLE,
            'pinecone_configured': bool(settings.PINECONE_API_KEY)
        }
        
        if self.mode == "local":
            status['vector_count'] = len(self.local_vectors)
        elif self.mode == "pinecone":
            try:
                stats = self.pinecone_index.describe_index_stats()
                status['vector_count'] = stats.total_vector_count
            except:
                status['vector_count'] = 'unknown'
        
        return status


# Global vector store instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """
    Get the global vector store instance.
    
    Returns:
        VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def upsert_embedding(key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
    """
    Store an embedding vector (convenience function).
    
    Args:
        key: Unique identifier
        vector: Embedding vector
        metadata: Associated metadata
    """
    store = get_vector_store()
    store.upsert_embedding(key, vector, metadata)


def query_similar(vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find similar vectors (convenience function).
    
    Args:
        vector: Query vector
        top_k: Number of results
        
    Returns:
        List of similar items
    """
    store = get_vector_store()
    return store.query_similar(vector, top_k)


def get_store_status() -> Dict[str, Any]:
    """
    Get vector store status (convenience function).
    
    Returns:
        Status information
    """
    store = get_vector_store()
    return store.get_status()
