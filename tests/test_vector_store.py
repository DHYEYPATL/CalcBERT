"""
Unit tests for Vector Store - v2 Backend
Tests local fallback vector store functionality.
"""

import pytest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.vector_store import VectorStore
from backend.config import settings


class TestVectorStore:
    """Test suite for vector store."""
    
    @pytest.fixture
    def temp_fallback_path(self):
        """Create temporary directory for fallback storage."""
        temp_dir = tempfile.mkdtemp()
        original_path = settings.VECTOR_FALLBACK_PATH
        settings.VECTOR_FALLBACK_PATH = os.path.join(temp_dir, "test_vectors.pkl")
        
        yield settings.VECTOR_FALLBACK_PATH
        
        # Cleanup
        settings.VECTOR_FALLBACK_PATH = original_path
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def vector_store(self, temp_fallback_path):
        """Create vector store instance."""
        # Force local mode for testing
        original_use_pinecone = settings.USE_PINECONE
        settings.USE_PINECONE = False
        
        store = VectorStore()
        
        yield store
        
        # Restore
        settings.USE_PINECONE = original_use_pinecone
    
    def test_initialization_local_mode(self, vector_store):
        """Test vector store initializes in local mode."""
        assert vector_store.mode == "local"
        assert vector_store.local_vectors == []
        assert vector_store.local_metadata == []
    
    def test_upsert_single_vector(self, vector_store):
        """Test upserting a single vector."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {"merchant": "Swiggy", "amount": 1200.0, "category": "Food"}
        
        vector_store.upsert_embedding("test_1", vector, metadata)
        
        assert len(vector_store.local_vectors) == 1
        assert len(vector_store.local_metadata) == 1
        assert vector_store.local_metadata[0]["merchant"] == "Swiggy"
    
    def test_upsert_multiple_vectors(self, vector_store):
        """Test upserting multiple vectors."""
        vectors = [
            ([0.1, 0.2, 0.3], {"merchant": "Swiggy", "category": "Food"}),
            ([0.4, 0.5, 0.6], {"merchant": "Uber", "category": "Transport"}),
            ([0.7, 0.8, 0.9], {"merchant": "Amazon", "category": "Shopping"})
        ]
        
        for i, (vec, meta) in enumerate(vectors):
            vector_store.upsert_embedding(f"test_{i}", vec, meta)
        
        assert len(vector_store.local_vectors) == 3
        assert len(vector_store.local_metadata) == 3
    
    def test_upsert_update_existing(self, vector_store):
        """Test updating an existing vector."""
        key = "test_update"
        vector1 = [0.1, 0.2, 0.3]
        metadata1 = {"merchant": "Swiggy", "amount": 1200.0}
        
        vector_store.upsert_embedding(key, vector1, metadata1)
        assert len(vector_store.local_vectors) == 1
        
        # Update with new data
        vector2 = [0.4, 0.5, 0.6]
        metadata2 = {"merchant": "Swiggy", "amount": 1500.0}
        
        vector_store.upsert_embedding(key, vector2, metadata2)
        
        # Should still have only 1 vector (updated)
        assert len(vector_store.local_vectors) == 1
        assert vector_store.local_metadata[0]["amount"] == 1500.0
    
    def test_query_similar_single_result(self, vector_store):
        """Test querying for similar vectors."""
        # Add some vectors
        vector_store.upsert_embedding("v1", [1.0, 0.0, 0.0], {"name": "v1"})
        vector_store.upsert_embedding("v2", [0.0, 1.0, 0.0], {"name": "v2"})
        vector_store.upsert_embedding("v3", [0.0, 0.0, 1.0], {"name": "v3"})
        
        # Query with vector similar to v1
        query_vector = [0.9, 0.1, 0.0]
        results = vector_store.query_similar(query_vector, top_k=1)
        
        assert len(results) == 1
        assert results[0]["metadata"]["name"] == "v1"
        assert results[0]["score"] > 0.8  # Should be high similarity
    
    def test_query_similar_multiple_results(self, vector_store):
        """Test querying for multiple similar vectors."""
        # Add vectors
        for i in range(5):
            vector = [float(i), 0.0, 0.0]
            vector_store.upsert_embedding(f"v{i}", vector, {"index": i})
        
        # Query
        query_vector = [2.5, 0.0, 0.0]
        results = vector_store.query_similar(query_vector, top_k=3)
        
        assert len(results) == 3
        # Results should be ordered by similarity
        assert results[0]["score"] >= results[1]["score"]
        assert results[1]["score"] >= results[2]["score"]
    
    def test_query_empty_store(self, vector_store):
        """Test querying an empty store."""
        query_vector = [1.0, 0.0, 0.0]
        results = vector_store.query_similar(query_vector, top_k=5)
        
        assert results == []
    
    def test_persistence(self, vector_store, temp_fallback_path):
        """Test that vectors are persisted to disk."""
        # Add vectors
        vector_store.upsert_embedding("v1", [1.0, 0.0], {"name": "v1"})
        vector_store.upsert_embedding("v2", [0.0, 1.0], {"name": "v2"})
        
        # Check file exists
        assert os.path.exists(temp_fallback_path)
        
        # Create new store instance (should load from disk)
        new_store = VectorStore()
        
        assert len(new_store.local_vectors) == 2
        assert len(new_store.local_metadata) == 2
    
    def test_get_status(self, vector_store):
        """Test getting store status."""
        status = vector_store.get_status()
        
        assert status["mode"] == "local"
        assert "vector_count" in status
        assert status["vector_count"] == 0
        
        # Add vectors
        vector_store.upsert_embedding("v1", [1.0, 0.0], {"name": "v1"})
        
        status = vector_store.get_status()
        assert status["vector_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
