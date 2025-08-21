import ollama
import logging
import os
import json
import pickle
import hashlib
import unittest
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import chromadb
from chromadb.config import Settings
import streamlit as st
import re
from collections import Counter
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration class for RAG system parameters."""
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    language_model: str = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
    chunk_size: int = 400  # Reduced for better granularity
    chunk_overlap: int = 100  # Increased overlap
    similarity_threshold: float = 0.3  # Lower threshold for more recall
    top_k: int = 5  # Increased for better coverage
    batch_size: int = 32
    cache_dir: str = "cache"
    use_vector_db: bool = True
    vector_db_path: str = "vector_db"
    rerank_results: bool = True  # New: Enable reranking
    use_hybrid_search: bool = True  # New: Enable hybrid search
    keyword_weight: float = 0.3  # New: Weight for keyword matching
    semantic_weight: float = 0.7  # New: Weight for semantic similarity
    min_chunk_overlap_tokens: int = 20  # New: Minimum overlap in tokens
    
    @classmethod
    def from_file(cls, config_path: str) -> 'RAGConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default config.")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return cls()
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

class QueryProcessor:
    """Enhanced query processing with expansion and normalization."""
    
    def __init__(self):
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'through', 'during', 'before', 'after', 'above', 'below',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
        
        # Academic synonyms for better matching
        self.synonyms = {
            'course': ['program', 'degree', 'study', 'curriculum', 'major'],
            'fee': ['cost', 'tuition', 'price', 'charge', 'expense'],
            'duration': ['length', 'time', 'period', 'semester', 'year'],
            'requirement': ['prerequisite', 'criteria', 'condition', 'qualification'],
            'career': ['job', 'profession', 'employment', 'work', 'opportunity'],
            'admission': ['enrollment', 'application', 'entry', 'acceptance'],
            'master': ['msc', 'ma', 'mba', 'graduate'],
            'bachelor': ['bsc', 'ba', 'undergraduate', 'degree']
        }
    
    def normalize_query(self, query: str) -> str:
        """Normalize query text."""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespaces
        query = re.sub(r'\s+', ' ', query)
        
        # Remove special characters except important ones
        query = re.sub(r'[^\w\s\-\.]', ' ', query)
        
        return query
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        normalized_query = self.normalize_query(query)
        words = normalized_query.split()
        
        # Remove stopwords
        keywords = [word for word in words if word not in self.stopwords and len(word) > 2]
        
        # Add synonyms
        expanded_keywords = keywords.copy()
        for keyword in keywords:
            if keyword in self.synonyms:
                expanded_keywords.extend(self.synonyms[keyword])
        
        return list(set(expanded_keywords))
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        keywords = self.extract_keywords(query)
        expanded_terms = []
        
        for keyword in keywords:
            expanded_terms.append(keyword)
            if keyword in self.synonyms:
                expanded_terms.extend(self.synonyms[keyword])
        
        # Create expanded query
        expanded_query = f"{query} {' '.join(set(expanded_terms))}"
        return expanded_query

class DocumentChunker:
    """Enhanced document chunking with better boundary detection."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def smart_chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Smart chunking with metadata and better boundary detection."""
        if len(text) <= self.chunk_size:
            return [{'content': text.strip(), 'start_idx': 0, 'end_idx': len(text)}] if text.strip() else []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Smart boundary detection
            if end < len(text):
                # Priority 1: Look for paragraph breaks
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 1
                else:
                    # Priority 2: Look for sentence endings
                    sentence_end = max([
                        text.rfind('. ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    ])
                    if sentence_end > start + self.chunk_size // 2:
                        end = sentence_end + 1
                    else:
                        # Priority 3: Look for comma or other punctuation
                        punct_pos = max([
                            text.rfind(', ', start, end),
                            text.rfind('; ', start, end),
                            text.rfind(': ', start, end)
                        ])
                        if punct_pos > start + self.chunk_size // 2:
                            end = punct_pos + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_info = {
                    'content': chunk_text,
                    'start_idx': start,
                    'end_idx': end,
                    'chunk_id': chunk_id,
                    'word_count': len(chunk_text.split())
                }
                chunks.append(chunk_info)
                chunk_id += 1
            
            # Calculate next start with overlap
            start = max(start + 1, end - self.chunk_overlap)
            if start >= len(text):
                break
                
        return chunks
    
    def chunk_document(self, content: str) -> List[Dict[str, Any]]:
        """Process entire document into smart chunks."""
        # Pre-process content
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Normalize paragraph breaks
        content = re.sub(r' +', ' ', content)  # Normalize spaces
        
        all_chunks = []
        
        # Try to identify sections (headers, etc.)
        sections = self.identify_sections(content)
        
        if sections:
            for section in sections:
                section_chunks = self.smart_chunk_text(section['content'])
                for chunk in section_chunks:
                    chunk['section'] = section['title']
                all_chunks.extend(section_chunks)
        else:
            all_chunks = self.smart_chunk_text(content)
        
        return all_chunks
    
    def identify_sections(self, content: str) -> List[Dict[str, str]]:
        """Identify document sections based on headers."""
        sections = []
        lines = content.split('\n')
        current_section = {'title': 'Introduction', 'content': ''}
        
        for line in lines:
            line = line.strip()
            
            # Check if line looks like a header (all caps, short, contains keywords)
            if (len(line) < 100 and 
                (line.isupper() or 
                 any(keyword in line.lower() for keyword in ['program', 'course', 'degree', 'faculty', 'department', 'school']))):
                
                # Save current section
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {'title': line, 'content': ''}
            else:
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections if len(sections) > 1 else []

class HybridRetriever:
    """Combines semantic and keyword-based retrieval."""
    
    def __init__(self, semantic_weight: float = 0.7, keyword_weight: float = 0.3):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.query_processor = QueryProcessor()
    
    def keyword_similarity(self, query: str, text: str) -> float:
        """Calculate keyword-based similarity using TF-IDF-like approach."""
        query_keywords = set(self.query_processor.extract_keywords(query))
        text_keywords = set(self.query_processor.extract_keywords(text))
        
        if not query_keywords:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_keywords.intersection(text_keywords)
        union = query_keywords.union(text_keywords)
        
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        
        # Calculate coverage (how many query keywords are covered)
        coverage = len(intersection) / len(query_keywords)
        
        # Combine Jaccard and coverage
        return (jaccard_sim + coverage) / 2
    
    def fuzzy_match_score(self, query: str, text: str) -> float:
        """Calculate fuzzy matching score for key terms."""
        query_normalized = self.query_processor.normalize_query(query)
        text_normalized = self.query_processor.normalize_query(text)
        
        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, query_normalized, text_normalized).ratio()
    
    def hybrid_score(self, query: str, text: str, semantic_score: float) -> float:
        """Combine semantic and keyword scores."""
        keyword_score = self.keyword_similarity(query, text)
        fuzzy_score = self.fuzzy_match_score(query, text)
        
        # Weighted combination
        final_score = (
            self.semantic_weight * semantic_score +
            self.keyword_weight * keyword_score +
            0.1 * fuzzy_score  # Small boost for fuzzy matches
        )
        
        return min(final_score, 1.0)  # Cap at 1.0

class EmbeddingCache:
    """Enhanced caching with better error handling."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.cache = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load cache from disk with better error handling."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded embedding cache with {len(cache)} entries")
                return cache
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            # Try to backup corrupted cache
            try:
                corrupted_file = self.cache_file.with_suffix('.pkl.corrupted')
                self.cache_file.rename(corrupted_file)
                logger.info(f"Moved corrupted cache to {corrupted_file}")
            except:
                pass
        return {}
    
    def _save_cache(self) -> None:
        """Save cache with atomic write."""
        try:
            temp_file = self.cache_file.with_suffix('.pkl.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(self.cache, f)
            temp_file.replace(self.cache_file)  # Atomic rename
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def _get_hash(self, text: str, model: str) -> str:
        """Generate hash for text and model combination."""
        # Include text length to avoid hash collisions
        return hashlib.md5(f"{text}:{model}:{len(text)}".encode()).hexdigest()
    
    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._get_hash(text, model)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def store_embedding(self, text: str, model: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        key = self._get_hash(text, model)
        self.cache[key] = embedding
        
        # Save cache every 10 new embeddings
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'total_entries': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.cache_hits = 0
        self.cache_misses = 0

class VectorDatabase:
    """Enhanced vector database with metadata support."""
    
    def __init__(self, db_path: str = "vector_db", collection_name: str = "documents"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection '{collection_name}'")
        except:
            self.collection = self.client.create_collection(
                collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection '{collection_name}'")
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     ids: List[str], metadatas: List[Dict] = None) -> None:
        """Add documents with metadata to the vector database."""
        if metadatas is None:
            metadatas = [{"chunk_id": i} for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], n_results: int = 5,
              where_clause: Dict = None) -> Dict[str, List]:
        """Search with optional filtering."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "distances", "metadatas"]
            )
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {"document_count": count}
        except:
            return {"document_count": 0}
    
    def clear(self) -> None:
        """Clear the database."""
        try:
            self.client.reset()
            logger.info("Vector database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")

class RAGSystem:
    """Enhanced RAG system with robust retrieval and response generation."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunker = DocumentChunker(config.chunk_size, config.chunk_overlap)
        self.cache = EmbeddingCache(config.cache_dir)
        self.query_processor = QueryProcessor()
        self.hybrid_retriever = HybridRetriever(
            config.semantic_weight, 
            config.keyword_weight
        ) if config.use_hybrid_search else None
        
        # Initialize vector database or in-memory storage
        if config.use_vector_db:
            try:
                self.vector_db = VectorDatabase(config.vector_db_path)
                self.use_vector_db = True
                logger.info("Using ChromaDB vector database")
            except Exception as e:
                logger.warning(f"Could not initialize ChromaDB: {e}. Falling back to in-memory storage.")
                self.vector_db = []
                self.use_vector_db = False
        else:
            self.vector_db = []
            self.use_vector_db = False
        
        self.document_chunks = []  # Store chunk metadata
        self.conversation_history = []  # Store conversation context
    
    def load_dataset(self, file_path: str) -> str:
        """Load dataset with better encoding handling."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as file:
                        content = file.read()
                    logger.info(f"Successfully loaded file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError("Could not decode file with any supported encoding")
                
            logger.info(f"Dataset loaded from {file_path} ({len(content)} characters)")
            return content
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Enhanced batch embedding with retry logic."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            if not text.strip():  # Skip empty texts
                embeddings.append([0.0] * 384)  # Default embedding dimension
                continue
                
            cached_embedding = self.cache.get_embedding(text, self.config.embedding_model)
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Get embeddings for uncached texts with retry logic
        if uncached_texts:
            try:
                batch_size = self.config.batch_size
                for i in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[i:i + batch_size]
                    batch_indices = uncached_indices[i:i + batch_size]
                    
                    for j, text in enumerate(batch):
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                response = ollama.embed(model=self.config.embedding_model, input=text)
                                embedding = response['embeddings'][0]
                                embeddings[batch_indices[j]] = embedding
                                self.cache.store_embedding(text, self.config.embedding_model, embedding)
                                break
                            except Exception as e:
                                if retry == max_retries - 1:
                                    logger.error(f"Failed to get embedding after {max_retries} retries: {e}")
                                    # Use zero embedding as fallback
                                    embeddings[batch_indices[j]] = [0.0] * 384
                                else:
                                    logger.warning(f"Retry {retry + 1} for embedding: {e}")
                        
            except Exception as e:
                logger.error(f"Error getting embeddings: {e}")
                raise
        
        return embeddings
    
    def build_database(self, content: str) -> None:
        """Build enhanced database with metadata."""
        logger.info("Processing document with smart chunking...")
        chunk_dicts = self.chunker.chunk_document(content)
        logger.info(f"Created {len(chunk_dicts)} smart chunks")
        
        if not chunk_dicts:
            raise ValueError("No chunks created from the document")
        
        # Extract chunk texts and metadata
        chunks = [chunk_dict['content'] for chunk_dict in chunk_dicts]
        self.document_chunks = chunk_dicts
        
        logger.info("Generating embeddings with caching...")
        embeddings = self.get_embedding_batch(chunks)
        
        if self.use_vector_db:
            # Prepare metadata for ChromaDB
            ids = [f"chunk_{chunk_dict['chunk_id']}" for chunk_dict in chunk_dicts]
            metadatas = [
                {
                    'chunk_id': chunk_dict['chunk_id'],
                    'word_count': chunk_dict['word_count'],
                    'section': chunk_dict.get('section', 'main'),
                    'start_idx': chunk_dict['start_idx'],
                    'end_idx': chunk_dict['end_idx']
                }
                for chunk_dict in chunk_dicts
            ]
            
            self.vector_db.add_documents(chunks, embeddings, ids, metadatas)
            logger.info(f"Vector database built with {len(chunks)} chunks")
            
            # Print database stats
            stats = self.vector_db.get_collection_stats()
            logger.info(f"Database stats: {stats}")
        else:
            # Store in memory with metadata
            self.vector_db = list(zip(chunks, embeddings, chunk_dicts))
            logger.info(f"In-memory database built with {len(chunks)} chunks")
        
        # Print cache stats
        cache_stats = self.cache.get_stats()
        logger.info(f"Cache stats: {cache_stats}")
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Enhanced cosine similarity calculation."""
        try:
            a_np = np.array(a, dtype=np.float32)
            b_np = np.array(b, dtype=np.float32)
            
            # Handle edge cases
            if len(a_np) != len(b_np):
                return 0.0
            
            dot_product = np.dot(a_np, b_np)
            norm_a = np.linalg.norm(a_np)
            norm_b = np.linalg.norm(b_np)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            similarity = float(dot_product / (norm_a * norm_b))
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def rerank_results(self, query: str, results: List[Tuple[str, float, Dict]]) -> List[Tuple[str, float, Dict]]:
        """Rerank results based on query relevance."""
        if not self.config.rerank_results or not results:
            return results
        
        reranked_results = []
        
        for chunk, similarity, metadata in results:
            # Calculate additional relevance signals
            query_keywords = set(self.query_processor.extract_keywords(query))
            chunk_keywords = set(self.query_processor.extract_keywords(chunk))
            
            # Keyword coverage
            keyword_coverage = len(query_keywords.intersection(chunk_keywords)) / len(query_keywords) if query_keywords else 0
            
            # Length penalty (prefer chunks that are not too short or too long)
            word_count = metadata.get('word_count', len(chunk.split()))
            length_penalty = 1.0
            if word_count < 20:
                length_penalty = 0.8  # Penalize very short chunks
            elif word_count > 200:
                length_penalty = 0.9  # Slightly penalize very long chunks
            
            # Section boost (if query mentions specific terms, boost relevant sections)
            section_boost = 1.0
            section = metadata.get('section', '').lower()
            if any(term in section for term in self.query_processor.extract_keywords(query)):
                section_boost = 1.2
            
            # Calculate final rerank score
            rerank_score = similarity * 0.7 + keyword_coverage * 0.3
            rerank_score *= length_penalty * section_boost
            
            reranked_results.append((chunk, rerank_score, metadata))
        
        # Sort by reranked score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results
    
    def retrieve(self, query: str) -> List[Tuple[str, float, Dict]]:
        """Enhanced retrieval with hybrid search and reranking."""
        # Sanitize and process query
        query = query.strip()
        if not query:
            return []
        
        try:
            # Expand query if needed
            if self.hybrid_retriever:
                expanded_query = self.query_processor.expand_query(query)
                logger.debug(f"Expanded query: {expanded_query}")
            else:
                expanded_query = query
            
            # Get query embedding
            query_embedding = self.get_embedding_batch([query])[0]
            
            if self.use_vector_db:
                # Use ChromaDB with expanded search
                search_results = self.vector_db.search(
                    query_embedding, 
                    self.config.top_k * 2  # Get more results for reranking
                )
                
                similarities = []
                for i, doc in enumerate(search_results['documents'][0]):
                    if not doc.strip():
                        continue
                        
                    # Calculate semantic similarity
                    distance = search_results['distances'][0][i]
                    semantic_similarity = max(0.0, 1.0 - distance)  # Convert distance to similarity
                    
                    metadata = search_results['metadatas'][0][i] if search_results['metadatas'][0] else {}
                    
                    # Apply hybrid scoring if enabled
                    if self.hybrid_retriever:
                        final_similarity = self.hybrid_retriever.hybrid_score(
                            expanded_query, doc, semantic_similarity
                        )
                    else:
                        final_similarity = semantic_similarity
                    
                    if final_similarity >= self.config.similarity_threshold:
                        similarities.append((doc, final_similarity, metadata))
                
            else:
                # Use in-memory storage
                similarities = []
                for chunk, embedding, metadata in self.vector_db:
                    semantic_similarity = self.cosine_similarity(query_embedding, embedding)
                    
                    # Apply hybrid scoring if enabled
                    if self.hybrid_retriever:
                        final_similarity = self.hybrid_retriever.hybrid_score(
                            expanded_query, chunk, semantic_similarity
                        )
                    else:
                        final_similarity = semantic_similarity
                    
                    if final_similarity >= self.config.similarity_threshold:
                        similarities.append((chunk, final_similarity, metadata))
            
            # Apply reranking
            similarities = self.rerank_results(query, similarities)
            
            # Return top results
            return similarities[:self.config.top_k]
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def generate_context_aware_response(self, query: str, retrieved_knowledge: List[Tuple[str, float, Dict]]) -> None:
        """Generate response with better context awareness and fact-checking."""
        if not retrieved_knowledge:
            print("No relevant information found in the database. Please try rephrasing your question or ask about topics covered in the SRH University documentation.")
            return
        
        # Prepare context with metadata
        context_parts = []
        total_confidence = 0
        
        for i, (chunk, similarity, metadata) in enumerate(retrieved_knowledge):
            section_info = f" (Section: {metadata.get('section', 'Main')})" if metadata.get('section') else ""
            confidence_indicator = "🔥" if similarity > 0.8 else "⭐" if similarity > 0.6 else "📝"
            
            context_parts.append(f"{confidence_indicator} Source {i+1}{section_info} (Confidence: {similarity:.2f}):\n{chunk}")
            total_confidence += similarity
        
        avg_confidence = total_confidence / len(retrieved_knowledge)
        context = '\n\n'.join(context_parts)
        
        # Add conversation history for context
        conversation_context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            conversation_context = "Previous conversation context:\n" + "\n".join([
                f"Q: {item['query']}\nA: {item['response'][:200]}..." for item in recent_history
            ]) + "\n\n"
        
        # Enhanced instruction prompt with better guidelines
        instruction_prompt = f'''You are an expert academic advisor for SRH University. Your role is to provide accurate, helpful information about SRH University's courses, programs, and services based ONLY on the provided context.

CONTEXT INFORMATION:
Average Confidence Level: {avg_confidence:.2f}
Number of Sources: {len(retrieved_knowledge)}

{conversation_context}CURRENT CONTEXT:
{context}

STRICT GUIDELINES:
1. ACCURACY FIRST: Only provide information that is explicitly stated in the context above
2. SRH FOCUS: All responses must be about SRH University only - do not mention other institutions
3. NO GUESSING: If information is not in the context, clearly state "This information is not available in my current knowledge base"
4. STRUCTURED RESPONSES: Organize information clearly with bullet points or numbered lists when appropriate
5. CONFIDENCE INDICATORS: When confidence is low (<0.5), mention "Based on available information..." 
6. COMPLETENESS: Address all parts of the question if possible
7. PRACTICAL FOCUS: Provide actionable information for students
8. FEES: Report semester fees as stated, don't calculate annual amounts
9. REQUIREMENTS: Be specific about admission requirements and prerequisites
10. CURRENCY: Always mention if fee information needs to be verified for current academic year

RESPONSE STYLE:
- Professional yet friendly and approachable
- Use clear, concise language appropriate for students
- Include relevant details like duration, fees, requirements
- Suggest next steps when appropriate (e.g., "Contact admissions for more details")
- If multiple options exist, compare them briefly

CURRENT STUDENT QUESTION: {query}

Provide a comprehensive answer based solely on the context above. If the context doesn't fully answer the question, acknowledge what information is missing and suggest how the student can get complete information.'''

        try:
            # Add query to conversation history before generating response
            current_exchange = {"query": query, "response": ""}
            
            stream = ollama.chat(
                model=self.config.language_model,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': query},
                ],
                stream=True,
            )
            
            print(f'\n🤖 SRH Academic Advisor (Confidence: {avg_confidence:.1%}):')
            print('-' * 60)
            
            response_parts = []
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                response_parts.append(content)
            
            # Store complete response in history
            full_response = ''.join(response_parts)
            current_exchange["response"] = full_response
            self.conversation_history.append(current_exchange)
            
            # Keep only recent history to avoid context overflow
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            print('\n' + '-' * 60)
            
            # Show source information
            print(f"\n📚 Information sourced from {len(retrieved_knowledge)} relevant document sections")
            if avg_confidence < 0.5:
                print("⚠️  Note: Lower confidence results - consider rephrasing your question for better accuracy")
            
            print()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            print("❌ Sorry, I encountered an error while generating the response. Please try again.")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'config': asdict(self.config),
            'cache_stats': self.cache.get_stats(),
            'conversation_history_length': len(self.conversation_history),
            'chunks_count': len(self.document_chunks)
        }
        
        if self.use_vector_db:
            stats['vector_db_stats'] = self.vector_db.get_collection_stats()
        else:
            stats['in_memory_db_size'] = len(self.vector_db)
        
        return stats
    
    def chat(self) -> None:
        """Enhanced interactive chat loop with better commands."""
        print("🎓 Enhanced SRH University RAG Chatbot initialized!")
        print("=" * 70)
        print("Available commands:")
        print("  'quit'/'exit'/'q'    - Exit the chatbot")
        print("  'config'             - Show current configuration")
        print("  'stats'              - Show detailed system statistics") 
        print("  'clear_cache'        - Clear embedding cache")
        print("  'clear_history'      - Clear conversation history")
        print("  'help'               - Show this help message")
        print("=" * 70)
        
        while True:
            try:
                query = input('\n🎯 Ask me about SRH University: ').strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using SRH University Academic Advisor. Goodbye!")
                    break
                    
                elif query.lower() == 'help':
                    print("\n📖 SRH University Academic Advisor Help:")
                    print("- Ask about courses, programs, fees, requirements")
                    print("- Ask about specific departments or faculties") 
                    print("- Ask about admission procedures and deadlines")
                    print("- Ask about campus facilities and student services")
                    print("- Use specific terms for better results (e.g., 'MBA fees' instead of 'cost')")
                    continue
                    
                elif query.lower() == 'config':
                    print("\n⚙️  Current Configuration:")
                    config_dict = asdict(self.config)
                    for key, value in config_dict.items():
                        print(f"  {key}: {value}")
                    continue
                    
                elif query.lower() == 'stats':
                    print("\n📊 System Statistics:")
                    stats = self.get_statistics()
                    for key, value in stats.items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for sub_key, sub_value in value.items():
                                print(f"    {sub_key}: {sub_value}")
                        else:
                            print(f"  {key}: {value}")
                    continue
                    
                elif query.lower() == 'clear_cache':
                    self.cache.clear_cache()
                    print("🧹 Embedding cache cleared.")
                    continue
                    
                elif query.lower() == 'clear_history':
                    self.conversation_history.clear()
                    print("🧹 Conversation history cleared.")
                    continue
                    
                if not query:
                    print("❓ Please enter a valid question about SRH University.")
                    continue
                
                print(f"\n🔍 Searching for: '{query}'")
                
                # Retrieve relevant knowledge with timing
                import time
                start_time = time.time()
                retrieved_knowledge = self.retrieve(query)
                retrieval_time = time.time() - start_time
                
                if retrieved_knowledge:
                    print(f"✅ Found {len(retrieved_knowledge)} relevant sources in {retrieval_time:.2f}s")
                    
                    # Show brief preview of sources
                    print("\n📋 Sources found:")
                    for i, (chunk, similarity, metadata) in enumerate(retrieved_knowledge, 1):
                        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
                        section = metadata.get('section', 'Main')
                        print(f"  {i}. [{section}] {similarity:.2f} - {preview}")
                else:
                    print("❌ No relevant information found")
                
                # Generate response
                self.generate_context_aware_response(query, retrieved_knowledge)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"❌ Sorry, an error occurred: {e}. Please try again.")

class TestRAGSystem(unittest.TestCase):
    """Enhanced unit tests for RAG system components."""
    
    def setUp(self):
        self.config = RAGConfig()
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        self.cache = EmbeddingCache("test_cache")
        self.query_processor = QueryProcessor()
        self.hybrid_retriever = HybridRetriever()
    
    def test_query_processor(self):
        """Test query processing functionality."""
        query = "What are the fees for MBA program?"
        
        # Test normalization
        normalized = self.query_processor.normalize_query(query)
        self.assertEqual(normalized, "what are the fees for mba program?")
        
        # Test keyword extraction
        keywords = self.query_processor.extract_keywords(query)
        self.assertIn("fees", keywords)
        self.assertIn("mba", keywords)
        self.assertIn("program", keywords)
        
        # Test query expansion
        expanded = self.query_processor.expand_query(query)
        self.assertIn("cost", expanded)  # Should include synonym
        self.assertIn("tuition", expanded)  # Should include synonym
    
    def test_smart_chunking(self):
        """Test enhanced document chunking."""
        text = "This is the first paragraph.\n\nThis is the second paragraph. " * 10
        chunks = self.chunker.smart_chunk_text(text)
        
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(isinstance(chunk, dict) for chunk in chunks))
        self.assertTrue(all('content' in chunk for chunk in chunks))
        self.assertTrue(all('chunk_id' in chunk for chunk in chunks))
    
    def test_hybrid_retrieval(self):
        """Test hybrid retrieval functionality."""
        query = "MBA program fees"
        text = "The Master of Business Administration (MBA) program costs 15000 euros per semester."
        
        # Test keyword similarity
        keyword_sim = self.hybrid_retriever.keyword_similarity(query, text)
        self.assertGreater(keyword_sim, 0)
        
        # Test hybrid scoring
        semantic_score = 0.8
        hybrid_score = self.hybrid_retriever.hybrid_score(query, text, semantic_score)
        self.assertGreater(hybrid_score, 0)
        self.assertLessEqual(hybrid_score, 1.0)
    
    def test_embedding_cache_enhanced(self):
        """Test enhanced embedding cache functionality."""
        test_text = "test text for caching"
        test_model = "test_model"
        test_embedding = [0.1, 0.2, 0.3]
        
        # Test storing and retrieving
        self.cache.store_embedding(test_text, test_model, test_embedding)
        retrieved = self.cache.get_embedding(test_text, test_model)
        
        self.assertEqual(retrieved, test_embedding)
        
        # Test statistics
        stats = self.cache.get_stats()
        self.assertIn('cache_hits', stats)
        self.assertIn('total_entries', stats)
        
        # Clean up
        self.cache.clear_cache()
    
    def test_cosine_similarity_enhanced(self):
        """Test enhanced cosine similarity calculation."""
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        vec3 = [1, 0, 0]
        vec4 = []  # Edge case
        vec5 = [0, 0, 0]  # Zero vector
        
        # Test normal cases
        self.assertAlmostEqual(RAGSystem.cosine_similarity(vec1, vec2), 0.0, places=5)
        self.assertAlmostEqual(RAGSystem.cosine_similarity(vec1, vec3), 1.0, places=5)
        
        # Test edge cases
        self.assertEqual(RAGSystem.cosine_similarity(vec1, vec4), 0.0)
        self.assertEqual(RAGSystem.cosine_similarity(vec1, vec5), 0.0)
    
    def test_config_management(self):
        """Test configuration management."""
        # Test default config
        config = RAGConfig()
        self.assertEqual(config.chunk_size, 400)
        self.assertEqual(config.similarity_threshold, 0.3)
        self.assertTrue(config.rerank_results)
        
        # Test config serialization
        config_dict = asdict(config)
        self.assertIn('embedding_model', config_dict)
        self.assertIn('rerank_results', config_dict)

def main():
    """Enhanced main function with better error handling."""
    # Load configuration
    config = RAGConfig.from_file("config.json")
    
    # Save default config if it doesn't exist
    config_path = Path("config.json")
    if not config_path.exists():
        config.save_to_file("config.json")
        print("✅ Created default config.json file. You can modify it to customize the system.")
    
    try:
        print("🚀 Initializing Enhanced RAG System...")
        
        # Initialize RAG system
        rag_system = RAGSystem(config)
        
        # Load and build database
        dataset_file = "university.txt"
        
        if not Path(dataset_file).exists():
            print(f"❌ Dataset file '{dataset_file}' not found!")
            print("📝 Please ensure 'university.txt' contains your SRH University documentation.")
            return
        
        print(f"📖 Loading dataset from {dataset_file}...")
        content = rag_system.load_dataset(dataset_file)
        
        print("🔨 Building enhanced database...")
        rag_system.build_database(content)
        
        print("✅ System initialization complete!")
        print(f"📊 Database contains {len(rag_system.document_chunks)} chunks")
        
        # Start interactive chat
        rag_system.chat()
        
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        print(f"❌ Error: {e}")
        print("📝 Please ensure 'university.txt' exists in the current directory.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        print(f"❌ Error: {e}")
        print("🔧 Troubleshooting tips:")
        print("  1. Ensure Ollama is running and accessible")
        print("  2. Check that the required models are installed:")
        print(f"     - {config.embedding_model}")
        print(f"     - {config.language_model}")
        print("  3. Verify 'university.txt' contains valid text data")
        print("  4. Check file permissions and disk space")

def run_tests():
    """Run comprehensive unit tests."""
    print("🧪 Running enhanced unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        main()

