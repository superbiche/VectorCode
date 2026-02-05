"""Tests for USearch database connector."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from usearch.index import Index

    usearch_available = True
except ImportError:
    usearch_available = False
    pytest.skip(
        "USearch not found. Skipping USearch tests.",
        allow_module_level=True,
    )

import numpy as np
from tree_sitter import Point

from vectorcode.chunking import Chunk
from vectorcode.cli_utils import Config
from vectorcode.database import types
from vectorcode.database.errors import CollectionNotFoundError
from vectorcode.database.usearch import MetadataDB, USearchConnector


@pytest.fixture
def mock_config():
    """Create a mock config with temporary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Config(
            project_root=tmpdir,
            embedding_function="SentenceTransformerEmbeddingFunction",
            db_type="USearchConnector",
            db_params={
                "db_path": os.path.join(tmpdir, "usearch_db"),
                "metric": "cos",
                "dtype": "f32",
                "connectivity": 16,
                "expansion_add": 128,
                "expansion_search": 64,
                "post_filter_multiplier": 10,
            },
        )


@pytest.fixture
def test_file(mock_config):
    """Create a test file in the project root."""
    file_path = os.path.join(mock_config.project_root, "test_file.py")
    with open(file_path, "w") as f:
        f.write("def hello():\n    print('hello world')\n")
    return file_path


class TestMetadataDB:
    """Tests for MetadataDB class."""

    def test_init_schema(self, mock_config):
        """Test database schema initialization."""
        db_path = Path(mock_config.db_params["db_path"]) / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        db = MetadataDB(db_path)
        db.init_schema()

        # Verify tables exist
        conn = db._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "collection_meta" in tables
        assert "documents" in tables

        db.close()

    def test_set_and_get_meta(self, mock_config):
        """Test setting and getting metadata."""
        db_path = Path(mock_config.db_params["db_path"]) / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        db = MetadataDB(db_path)
        db.init_schema()

        db.set_meta("test_key", "test_value")
        assert db.get_meta("test_key") == "test_value"
        assert db.get_meta("nonexistent") is None

        db.close()

    def test_add_and_get_chunks(self, mock_config):
        """Test adding and retrieving chunks."""
        db_path = Path(mock_config.db_params["db_path"]) / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        db = MetadataDB(db_path)
        db.init_schema()

        chunks = [
            {
                "id": 1,
                "path": "/test/file.py",
                "sha256": "hash1",
                "text": "chunk text 1",
                "line_start": 1,
                "line_end": 10,
            },
            {
                "id": 2,
                "path": "/test/file.py",
                "sha256": "hash1",
                "text": "chunk text 2",
                "line_start": 11,
                "line_end": 20,
            },
        ]
        db.add_chunks(chunks)

        # Test get_paths
        paths = db.get_paths([1, 2])
        assert paths == ["/test/file.py", "/test/file.py"]

        # Test get_chunks_by_ids
        retrieved = db.get_chunks_by_ids([1, 2])
        assert len(retrieved) == 2
        assert retrieved[0]["text"] == "chunk text 1"

        # Test get_all_unique_paths
        unique_paths = db.get_all_unique_paths()
        assert unique_paths == ["/test/file.py"]

        # Test count methods
        assert db.count_chunks() == 2
        assert db.count_files() == 1

        db.close()

    def test_delete_by_paths(self, mock_config):
        """Test deleting chunks by path."""
        db_path = Path(mock_config.db_params["db_path"]) / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        db = MetadataDB(db_path)
        db.init_schema()

        chunks = [
            {"id": 1, "path": "/file1.py", "sha256": "h1", "text": "t1", "line_start": 1, "line_end": 10},
            {"id": 2, "path": "/file2.py", "sha256": "h2", "text": "t2", "line_start": 1, "line_end": 10},
        ]
        db.add_chunks(chunks)

        deleted_ids = db.delete_by_paths(["/file1.py"])
        assert deleted_ids == [1]
        assert db.count_chunks() == 1

        db.close()

    def test_get_next_id(self, mock_config):
        """Test getting next available ID."""
        db_path = Path(mock_config.db_params["db_path"]) / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        db = MetadataDB(db_path)
        db.init_schema()

        # Empty database
        assert db.get_next_id() == 1

        chunks = [{"id": 5, "path": "/file.py", "sha256": "h", "text": "t", "line_start": None, "line_end": None}]
        db.add_chunks(chunks)
        assert db.get_next_id() == 6

        db.close()


class TestUSearchConnector:
    """Tests for USearchConnector class."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test connector initialization."""
        connector = USearchConnector(mock_config)
        assert connector._configs.project_root == mock_config.project_root
        assert connector._configs.db_params["metric"] == "cos"

    @pytest.mark.asyncio
    async def test_get_collection_dir(self, mock_config):
        """Test collection directory path generation."""
        connector = USearchConnector(mock_config)
        collection_dir = connector._get_collection_dir()
        assert str(collection_dir).startswith(mock_config.db_params["db_path"])

    @pytest.mark.asyncio
    async def test_vectorise(self, mock_config, test_file):
        """Test the vectorise method."""
        connector = USearchConnector(mock_config)

        # Mock chunker and embedding
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "def hello():"
        mock_chunk.start = Point(row=1, column=0)
        mock_chunk.end = Point(row=2, column=0)
        mock_chunker.chunk.return_value = [mock_chunk]

        # Mock embeddings
        connector.get_embedding = MagicMock(return_value=[np.random.randn(768).astype(np.float32)])

        with patch("vectorcode.database.usearch.hash_file", return_value="testhash"):
            stats = await connector.vectorise(test_file, chunker=mock_chunker)

        assert stats.add == 1

        # Verify collection was created
        collection_dir = connector._get_collection_dir()
        assert collection_dir.exists()
        assert (collection_dir / "index.usearch").exists()
        assert (collection_dir / "metadata.db").exists()

    @pytest.mark.asyncio
    async def test_vectorise_no_chunks(self, mock_config, test_file):
        """Test vectorise when file produces no chunks."""
        connector = USearchConnector(mock_config)

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []

        stats = await connector.vectorise(test_file, chunker=mock_chunker)
        assert stats.skipped == 1

    @pytest.mark.asyncio
    async def test_query(self, mock_config, test_file):
        """Test the query method."""
        connector = USearchConnector(mock_config)

        # First, add some data
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "def hello():"
        mock_chunk.start = Point(row=1, column=0)
        mock_chunk.end = Point(row=2, column=0)
        mock_chunker.chunk.return_value = [mock_chunk]

        embedding = np.random.randn(768).astype(np.float32)
        connector.get_embedding = MagicMock(return_value=[embedding])

        with patch("vectorcode.database.usearch.hash_file", return_value="testhash"):
            await connector.vectorise(test_file, chunker=mock_chunker)

        # Now query
        connector._configs.query = ["hello"]
        connector._configs.n_result = 1

        results = await connector.query()
        assert len(results) == 1
        assert results[0].path == test_file

    @pytest.mark.asyncio
    async def test_query_with_exclusion(self, mock_config, test_file):
        """Test query with path exclusion."""
        connector = USearchConnector(mock_config)

        # Add data
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "def hello():"
        mock_chunk.start = Point(row=1, column=0)
        mock_chunk.end = Point(row=2, column=0)
        mock_chunker.chunk.return_value = [mock_chunk]

        embedding = np.random.randn(768).astype(np.float32)
        connector.get_embedding = MagicMock(return_value=[embedding])

        with patch("vectorcode.database.usearch.hash_file", return_value="testhash"):
            await connector.vectorise(test_file, chunker=mock_chunker)

        # Query with exclusion
        connector._configs.query = ["hello"]
        connector._configs.n_result = 1
        connector._configs.query_exclude = [test_file]

        results = await connector.query()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_collection_not_found(self, mock_config):
        """Test query when collection doesn't exist."""
        connector = USearchConnector(mock_config)
        connector._configs.query = ["test"]

        with pytest.raises(CollectionNotFoundError):
            await connector.query()

    @pytest.mark.asyncio
    async def test_list_collections(self, mock_config, test_file):
        """Test listing collections."""
        connector = USearchConnector(mock_config)

        # Create a collection
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "test"
        mock_chunk.start = None
        mock_chunk.end = None
        mock_chunker.chunk.return_value = [mock_chunk]

        connector.get_embedding = MagicMock(return_value=[np.random.randn(768).astype(np.float32)])

        with patch("vectorcode.database.usearch.hash_file", return_value="testhash"):
            await connector.vectorise(test_file, chunker=mock_chunker)

        collections = await connector.list_collections()
        assert len(collections) == 1
        assert collections[0].database_backend == "USearch"
        assert collections[0].chunk_count == 1

    @pytest.mark.asyncio
    async def test_list_collection_content(self, mock_config, test_file):
        """Test listing collection content."""
        connector = USearchConnector(mock_config)

        # Add data
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "test chunk"
        mock_chunk.start = Point(row=1, column=0)
        mock_chunk.end = Point(row=5, column=0)
        mock_chunker.chunk.return_value = [mock_chunk]

        connector.get_embedding = MagicMock(return_value=[np.random.randn(768).astype(np.float32)])

        with patch("vectorcode.database.usearch.hash_file", return_value="testhash"):
            await connector.vectorise(test_file, chunker=mock_chunker)

        # Get all content
        content = await connector.list_collection_content()
        assert len(content.files) == 1
        assert len(content.chunks) == 1

        # Get only files
        content = await connector.list_collection_content(what=types.ResultType.document)
        assert len(content.files) == 1
        assert len(content.chunks) == 0

        # Get only chunks
        content = await connector.list_collection_content(what=types.ResultType.chunk)
        assert len(content.files) == 0
        assert len(content.chunks) == 1

    @pytest.mark.asyncio
    async def test_list_collection_content_not_found(self, mock_config):
        """Test list_collection_content when collection doesn't exist."""
        connector = USearchConnector(mock_config)

        with pytest.raises(CollectionNotFoundError):
            await connector.list_collection_content()

    @pytest.mark.asyncio
    async def test_delete(self, mock_config, test_file):
        """Test deleting files from collection."""
        connector = USearchConnector(mock_config)

        # Add data
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "test"
        mock_chunk.start = None
        mock_chunk.end = None
        mock_chunker.chunk.return_value = [mock_chunk]

        connector.get_embedding = MagicMock(return_value=[np.random.randn(768).astype(np.float32)])

        with patch("vectorcode.database.usearch.hash_file", return_value="testhash"):
            await connector.vectorise(test_file, chunker=mock_chunker)

        # Verify file exists
        content = await connector.list_collection_content(what=types.ResultType.document)
        assert len(content.files) == 1

        # Delete
        connector._configs.rm_paths = [test_file]
        with (
            patch("vectorcode.database.usearch.expand_globs", return_value=[test_file]),
            patch("vectorcode.database.usearch.expand_path", side_effect=lambda path, absolute: path),
        ):
            deleted = await connector.delete()
            assert deleted == 1

        # Verify file is gone from metadata
        content = await connector.list_collection_content(what=types.ResultType.document)
        assert len(content.files) == 0

    @pytest.mark.asyncio
    async def test_drop(self, mock_config, test_file):
        """Test dropping a collection."""
        connector = USearchConnector(mock_config)

        # Create collection
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "test"
        mock_chunk.start = None
        mock_chunk.end = None
        mock_chunker.chunk.return_value = [mock_chunk]

        connector.get_embedding = MagicMock(return_value=[np.random.randn(768).astype(np.float32)])

        with patch("vectorcode.database.usearch.hash_file", return_value="testhash"):
            await connector.vectorise(test_file, chunker=mock_chunker)

        collection_dir = connector._get_collection_dir()
        assert collection_dir.exists()

        # Drop
        await connector.drop()
        assert not collection_dir.exists()

    @pytest.mark.asyncio
    async def test_drop_not_found(self, mock_config):
        """Test dropping a non-existent collection."""
        connector = USearchConnector(mock_config)

        with pytest.raises(CollectionNotFoundError):
            await connector.drop()

    @pytest.mark.asyncio
    async def test_get_chunks(self, mock_config, test_file):
        """Test retrieving chunks for a file."""
        connector = USearchConnector(mock_config)

        # Add data
        mock_chunker = MagicMock()
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "chunk 1"
        mock_chunk1.start = Point(row=1, column=0)
        mock_chunk1.end = Point(row=5, column=0)
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "chunk 2"
        mock_chunk2.start = Point(row=6, column=0)
        mock_chunk2.end = Point(row=10, column=0)
        mock_chunker.chunk.return_value = [mock_chunk1, mock_chunk2]

        embeddings = [np.random.randn(768).astype(np.float32) for _ in range(2)]
        connector.get_embedding = MagicMock(return_value=embeddings)

        with patch("vectorcode.database.usearch.hash_file", return_value="testhash"):
            await connector.vectorise(test_file, chunker=mock_chunker)

        chunks = await connector.get_chunks(test_file)
        assert len(chunks) == 2
        assert chunks[0].text == "chunk 1"
        assert chunks[1].text == "chunk 2"

    @pytest.mark.asyncio
    async def test_get_chunks_not_found(self, mock_config, test_file):
        """Test get_chunks when collection doesn't exist."""
        connector = USearchConnector(mock_config)

        chunks = await connector.get_chunks(test_file)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_count(self, mock_config, test_file):
        """Test counting chunks and files."""
        connector = USearchConnector(mock_config)

        # Add data
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "test"
        mock_chunk.start = None
        mock_chunk.end = None
        mock_chunker.chunk.return_value = [mock_chunk]

        connector.get_embedding = MagicMock(return_value=[np.random.randn(768).astype(np.float32)])

        with patch("vectorcode.database.usearch.hash_file", return_value="testhash"):
            await connector.vectorise(test_file, chunker=mock_chunker)

        chunk_count = await connector.count(types.ResultType.chunk)
        assert chunk_count == 1

        file_count = await connector.count(types.ResultType.document)
        assert file_count == 1


class TestUSearchConnectorIntegration:
    """Integration tests that use real USearch operations."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_config):
        """Test complete indexing and querying workflow."""
        connector = USearchConnector(mock_config)

        # Create test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(mock_config.project_root, f"file_{i}.py")
            with open(file_path, "w") as f:
                f.write(f"# File {i}\ndef function_{i}():\n    pass\n")
            test_files.append(file_path)

        # Index files
        mock_chunker = MagicMock()

        for i, file_path in enumerate(test_files):
            mock_chunk = MagicMock()
            mock_chunk.text = f"function_{i}"
            mock_chunk.start = Point(row=2, column=0)
            mock_chunk.end = Point(row=3, column=0)
            mock_chunker.chunk.return_value = [mock_chunk]

            # Use consistent but different embeddings per file
            np.random.seed(i)
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            connector.get_embedding = MagicMock(return_value=[embedding])

            with patch("vectorcode.database.usearch.hash_file", return_value=f"hash_{i}"):
                await connector.vectorise(file_path, chunker=mock_chunker)

        # Verify indexing
        collections = await connector.list_collections()
        assert len(collections) == 1
        assert collections[0].chunk_count == 3
        assert collections[0].file_count == 3

        # Query
        connector._configs.query = ["function"]
        connector._configs.n_result = 3

        # Use a query embedding similar to file 0
        np.random.seed(0)
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        connector.get_embedding = MagicMock(return_value=[query_embedding])

        results = await connector.query()
        assert len(results) == 3

        # Query with exclusion
        connector._configs.query_exclude = [test_files[0]]
        results = await connector.query()
        assert len(results) == 2
        assert all(r.path != test_files[0] for r in results)

        # Delete a file
        connector._configs.rm_paths = [test_files[1]]
        with (
            patch("vectorcode.database.usearch.expand_globs", return_value=[test_files[1]]),
            patch("vectorcode.database.usearch.expand_path", side_effect=lambda path, absolute: path),
        ):
            deleted = await connector.delete()
            assert deleted == 1

        # Verify deletion
        content = await connector.list_collection_content(what=types.ResultType.document)
        assert len(content.files) == 2

        # Drop collection
        await connector.drop()
        collections = await connector.list_collections()
        assert len(collections) == 0

    @pytest.mark.asyncio
    async def test_post_filter_multiplier(self, mock_config):
        """Test that post-filter multiplier correctly handles exclusions."""
        connector = USearchConnector(mock_config)

        # Create many chunks to test post-filtering
        file_path = os.path.join(mock_config.project_root, "large_file.py")
        with open(file_path, "w") as f:
            f.write("# Large file\n" * 100)

        # Add 20 chunks
        mock_chunker = MagicMock()
        chunks = []
        for i in range(20):
            chunk = MagicMock()
            chunk.text = f"chunk_{i}"
            chunk.start = Point(row=i * 5, column=0)
            chunk.end = Point(row=(i + 1) * 5, column=0)
            chunks.append(chunk)
        mock_chunker.chunk.return_value = chunks

        np.random.seed(42)
        embeddings = [np.random.randn(768).astype(np.float32) for _ in range(20)]
        connector.get_embedding = MagicMock(return_value=embeddings)

        with patch("vectorcode.database.usearch.hash_file", return_value="hash"):
            await connector.vectorise(file_path, chunker=mock_chunker)

        # Query with n_result=5, should work with post-filter
        connector._configs.query = ["test"]
        connector._configs.n_result = 5
        connector._configs.query_exclude = []

        np.random.seed(42)
        query_embedding = np.random.randn(768).astype(np.float32)
        connector.get_embedding = MagicMock(return_value=[query_embedding])

        results = await connector.query()
        assert len(results) == 5
