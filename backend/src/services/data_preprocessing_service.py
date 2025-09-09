#!/usr/bin/env python3
"""
Data Preprocessing Service for Localized Translator using Docling
Uses Docling for robust document processing and text extraction
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import re
from dataclasses import dataclass
import json
from uuid import uuid4

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.models.base_model import DoclingDocument
from docling.chunking import HybridChunker

# Tokenizer for alignment with embedding model
from transformers import AutoTokenizer

# Configuration
from ..config import get_preprocessing_config

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    # Docling pipeline options
    enable_ocr: bool = True
    enable_table_extraction: bool = True
    enable_figure_extraction: bool = True
    enable_formula_extraction: bool = True
    
    # Docling chunking options
    use_docling_chunking: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    preserve_tables: bool = True  # Keep tables atomic
    
    # Text processing options
    min_text_length: int = 3
    quality_threshold: float = 0.3
    
    # Text cleaning options
    collapse_whitespace: bool = True
    normalize_quotes_dashes: bool = True
    strip_control_chars: bool = True
    preserve_diacritics: bool = True  # Don't remove Vietnamese diacritics
    force_lowercase: bool = False     # Don't force lowercase
    
    # Supported formats
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                'pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls',
                'csv', 'txt', 'md', 'html', 'xml', 'json'
            ]
    
    @classmethod
    def from_config(cls, config=None) -> 'ProcessingConfig':
        """Create ProcessingConfig from centralized config"""
        if config is None:
            config = get_preprocessing_config()
        
        return cls(
            enable_ocr=config.enable_ocr,
            enable_table_extraction=config.enable_table_extraction,
            enable_figure_extraction=config.enable_figure_extraction,
            enable_formula_extraction=config.enable_formula_extraction,
            use_docling_chunking=config.use_docling_chunking,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            preserve_tables=config.preserve_tables,
            min_text_length=config.min_text_length,
            quality_threshold=config.quality_threshold,
            collapse_whitespace=config.collapse_whitespace,
            normalize_quotes_dashes=config.normalize_quotes_dashes,
            strip_control_chars=config.strip_control_chars,
            preserve_diacritics=config.preserve_diacritics,
            force_lowercase=config.force_lowercase,
            supported_formats=config.supported_formats
        )

@dataclass
class ProcessedDocument:
    """Result of document processing"""
    file_path: str
    file_type: str
    content: str
    metadata: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    quality_score: float
    processing_time: float
    docling_document: DoclingDocument

class DoclingDataPreprocessingService:
    """Data preprocessing service using Docling for document processing"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        # Use centralized config if none provided
        if config is None:
            config = ProcessingConfig.from_config()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Docling converter and chunker (with aligned tokenizer)
        self.docling_converter = self._setup_docling_converter()
        self.docling_chunker = self._setup_docling_chunker()
        
        # Common noise patterns for text cleaning
        self.noise_patterns = [
            r'^[-\s]+$',        # Just dashes or spaces
            r'^[^\w\s]+$',      # Just punctuation
            r'^\([^)]*\)$',     # Just parentheses
            r'^[^\w\s]*$',      # No alphanumeric
        ]
        
        # Invalid content to filter out
        self.invalid_content = [
            '(no tagline)', '(image only)', '(no more SS images shown)',
            '#ERROR!', 'N/A', 'NULL', 'undefined', 'missing'
        ]
    
    def _setup_docling_converter(self) -> DocumentConverter:
        """Setup Docling document converter with appropriate options"""
        try:
            # Create converter with the correct constructor parameters
            converter = DocumentConverter(
                allowed_formats=[InputFormat(fmt) for fmt in self.config.supported_formats]
            )
            
            self.logger.info("✅ Docling converter initialized successfully")
            return converter
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Docling converter: {e}")
            raise
    
    def _setup_docling_chunker(self) -> HybridChunker:
        """Setup Docling hybrid chunker with aligned tokenizer"""
        try:
            # Get tokenizer model from config (automatically inferred from embedding model)
            preprocessing_config = get_preprocessing_config()
            tokenizer_model = preprocessing_config.tokenizer_model
            
            # Setup the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            
            # Create hybrid chunker with aligned tokenizer
            chunker = HybridChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                tokenizer=tokenizer  # Use the same tokenizer as embedding model
            )
            
            self.logger.info("✅ Docling hybrid chunker initialized with aligned tokenizer")
            self.logger.info(f"   Embedding model: {preprocessing_config.embedding_model}")
            self.logger.info(f"   Tokenizer model: {tokenizer_model} (inferred)")
            self.logger.info(f"   Chunk size: {self.config.chunk_size}")
            self.logger.info(f"   Chunk overlap: {self.config.chunk_overlap}")
            return chunker
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Docling chunker: {e}")
            raise
    

    
    def process_document(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """
        Process a single document using Docling
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            ProcessedDocument with extracted content and metadata
        """
        import time
        start_time = time.time()
        
        file_path = Path(file_path)
        
        try:
            self.logger.info(f"🔄 Processing document: {file_path.name}")
            
            # Check if file exists and is supported
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = file_path.suffix.lower().lstrip('.')
            if file_extension not in self.config.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Process with Docling
            docling_doc = self.docling_converter.convert(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata_from_docling(docling_doc, file_path)
            
            # Create chunks using Docling's built-in chunker - let Docling handle everything
            chunks = self._create_chunks(docling_doc, metadata)
            
            # Extract content for quality assessment (let Docling handle the extraction)
            content = self._extract_content_from_docling(docling_doc)
            cleaned_content = self._clean_text(content)
            quality_score = self._assess_content_quality(cleaned_content)
            
            processing_time = time.time() - start_time
            
            result = ProcessedDocument(
                file_path=str(file_path),
                file_type=file_extension,
                content=cleaned_content,
                metadata=metadata,
                chunks=chunks,
                quality_score=quality_score,
                processing_time=processing_time,
                docling_document=docling_doc
            )
            
            self.logger.info(f"✅ Document processed successfully:")
            self.logger.info(f"   Content length: {len(cleaned_content)} chars")
            self.logger.info(f"   Chunks created: {len(chunks)}")
            self.logger.info(f"   Quality score: {quality_score:.3f}")
            self.logger.info(f"   Processing time: {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Failed to process document {file_path.name}: {e}")
            raise
    
    def _extract_content_from_docling(self, docling_doc: DoclingDocument) -> str:
        """Extract text content from Docling document"""
        try:
            content_parts = []
            
            # For CSV files, use specialized extraction
            if hasattr(docling_doc, 'document') and docling_doc.document:
                doc = docling_doc.document
                
                # Check if this is a CSV file (has tables but no meaningful text content)
                if hasattr(doc, 'tables') and doc.tables and (not hasattr(doc, 'texts') or not doc.texts):
                    # This is likely a CSV file - extract table data properly
                    for table in doc.tables:
                        try:
                            # Use export_to_markdown for CSV handling (this is what works)
                            if hasattr(table, 'export_to_markdown'):
                                markdown_text = table.export_to_markdown()
                                # Clean and structure the markdown table
                                structured_text = self._clean_markdown_table(markdown_text)
                                content_parts.append(structured_text)
                        except Exception as e:
                            self.logger.warning(f"Failed to export table: {e}")
                            continue
                else:
                    # Regular document processing
                    if hasattr(doc, 'export_to_text'):
                        content_parts.append(doc.export_to_text())
                    elif hasattr(doc, 'texts') and doc.texts:
                        content_parts.extend(doc.texts)
                    
                    # Extract from tables if available
                    if hasattr(doc, 'tables') and doc.tables:
                        for table in doc.tables:
                            if hasattr(table, 'export_to_text'):
                                content_parts.append(table.export_to_text())
                            elif hasattr(table, 'texts') and table.texts:
                                content_parts.extend(table.texts)
            
            # Extract from figures/captions if available
            if hasattr(docling_doc.document, 'pictures') and docling_doc.document.pictures:
                for picture in docling_doc.document.pictures:
                    if hasattr(picture, 'caption') and picture.caption:
                        content_parts.append(picture.caption)
            
            # Combine all content
            full_content = "\n\n".join(filter(None, content_parts))
            
            # Fallback: try to get raw text if structured extraction fails
            if not full_content.strip() and hasattr(docling_doc.document, 'export_to_markdown'):
                try:
                    full_content = docling_doc.document.export_to_markdown()
                except:
                    pass
            
            return full_content or ""
            
        except Exception as e:
            self.logger.warning(f"Failed to extract content from Docling document: {e}")
            return ""
    
    def _clean_markdown_table(self, markdown_text: str) -> str:
        """Clean and structure markdown table for translation memory"""
        try:
            lines = markdown_text.strip().split('\n')
            structured_lines = []
            
            # Find header line (first line with data)
            header_line = None
            data_start_idx = 0
            
            for i, line in enumerate(lines):
                if line.strip() and '|' in line and not re.match(r'^[-\s|:]+$', line.strip()):
                    header_line = line
                    data_start_idx = i + 2  # Skip separator line
                    break
            
            if header_line:
                # Parse header to get column names
                headers = [col.strip() for col in header_line.split('|') if col.strip()]
                
                # Process data lines (skip separator line)
                for line in lines[data_start_idx:]:
                    if line.strip() and '|' in line and not re.match(r'^[-\s|:]+$', line.strip()):
                        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                        if len(cells) >= 2:  # At least 2 columns
                            # Create structured line
                            line_parts = []
                            for i, cell in enumerate(cells):
                                if cell and cell.strip() and i < len(headers):
                                    line_parts.append(f"{headers[i]}: {cell.strip()}")
                            
                            if line_parts:
                                structured_lines.append(" | ".join(line_parts))
            else:
                # Fallback: treat as simple table without headers
                for line in lines:
                    if line.strip() and '|' in line and not re.match(r'^[-\s|:]+$', line.strip()):
                        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                        if len(cells) >= 2:
                            line_parts = []
                            for i, cell in enumerate(cells):
                                if cell and cell.strip():
                                    line_parts.append(f"Column_{i+1}: {cell.strip()}")
                            
                            if line_parts:
                                structured_lines.append(" | ".join(line_parts))
            
            return "\n".join(structured_lines)
            
        except Exception as e:
            self.logger.warning(f"Failed to clean markdown table: {e}")
            return markdown_text
    
    def _extract_metadata_from_docling(self, docling_doc: DoclingDocument, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from Docling document"""
        metadata = {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'processing_timestamp': pd.Timestamp.now().isoformat(),
        }
        
        try:
            # Extract document-level metadata
            if hasattr(docling_doc, 'document') and docling_doc.document:
                doc = docling_doc.document
                
                # Add document properties
                if hasattr(doc, 'properties'):
                    metadata.update(doc.properties)
                
                # Add page count if available
                if hasattr(doc, 'pages'):
                    metadata['page_count'] = len(doc.pages)
                
                # Add table count
                if hasattr(docling_doc, 'tables'):
                    metadata['table_count'] = len(docling_doc.tables)
                
                # Add figure count
                if hasattr(docling_doc, 'figures'):
                    metadata['figure_count'] = len(docling_doc.figures)
            
            # Add processing confidence if available
            if hasattr(docling_doc, 'confidence_scores'):
                metadata['confidence_scores'] = docling_doc.confidence_scores
            
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata from Docling document: {e}")
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text according to specific requirements:
        - Collapse whitespace; normalize quotes/dashes; strip control chars
        - Do not remove Vietnamese diacritics or force lowercase
        """
        if not text or pd.isna(text):
            return ""
        
        # Basic cleaning
        cleaned = str(text).strip()
        
        # Remove invalid content
        if any(indicator.lower() in cleaned.lower() for indicator in self.invalid_content):
            return ""
        
        # Remove noise patterns
        for pattern in self.noise_patterns:
            if re.match(pattern, cleaned):
                return ""
        
        # 1. Collapse whitespace (multiple spaces/tabs/newlines to single space)
        if self.config.collapse_whitespace:
            cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 2. Normalize quotes and dashes
        if self.config.normalize_quotes_dashes:
            # Normalize quotes
            cleaned = re.sub(r'["""]', '"', cleaned)
            cleaned = re.sub(r"[''']", "'", cleaned)
            # Normalize dashes
            cleaned = re.sub(r'[–—]', '-', cleaned)
        
        # 3. Strip control characters (but preserve diacritics)
        if self.config.strip_control_chars:
            # Remove control characters but keep printable Unicode characters
            # This preserves Vietnamese diacritics and other accented characters
            cleaned = ''.join(char for char in cleaned if char.isprintable() or char.isspace())
        
        # 4. Do NOT remove Vietnamese diacritics (preserve_diacritics=True by default)
        # 5. Do NOT force lowercase (force_lowercase=False by default)
        
        # Final whitespace cleanup
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _generate_sparse_vector(self, text: str) -> Dict[str, Any]:
        """
        Generate proper sparse vector for Qdrant using SPLADE via FastEmbed
        
        Returns a dictionary with 'indices' and 'values' for Qdrant's SparseVector format
        """
        try:
            from fastembed import SparseTextEmbedding
            
            # Initialize SPLADE model for sparse embeddings
            sparse_model = SparseTextEmbedding("prithvida/Splade_PP_en_v1")
            
            # Generate sparse embedding
            sparse_embeddings = list(sparse_model.embed([text]))
            sparse_embedding = sparse_embeddings[0]
            
            # Convert to Qdrant's SparseVector format
            # SPLADE returns indices and values
            sparse_vector = {
                "indices": sparse_embedding.indices.tolist(),
                "values": sparse_embedding.values.tolist()
            }
            
            return sparse_vector
            
        except ImportError:
            self.logger.error("FastEmbed SPLADE not available - this is required for hybrid search")
            return {"indices": [], "values": []}
        except Exception as e:
            self.logger.error(f"Failed to generate SPLADE sparse vector: {e}")
            return {"indices": [], "values": []}
    

    
    
    def _create_chunks(self, docling_doc: DoclingDocument, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks using Docling's HybridChunker
        Preserves table atomicity and document structure
        """
        if not docling_doc or not docling_doc.document:
            return []
        
        chunks = []
        
        try:
            if self.config.use_docling_chunking:
                # Use HybridChunker with aligned tokenizer
                docling_chunks = list(self.docling_chunker.chunk(docling_doc.document))
                
                for i, chunk in enumerate(docling_chunks):
                    # Use Docling's chunk text directly - let Docling handle all processing
                    chunk_text = chunk.text
                    
                    if len(chunk_text) >= self.config.min_text_length:
                        # Create chunk metadata
                        chunk_metadata = metadata.copy()
                        # Count tokens using Docling's tokenizer
                        token_count = self.docling_chunker.tokenizer.count_tokens(chunk_text)
                        
                        chunk_metadata.update({
                            'chunk_index': i,
                            'total_chunks': len(docling_chunks),
                            'token_count': token_count,
                            'chunk_type': getattr(chunk, 'chunk_type', 'text'),
                            'is_table': getattr(chunk, 'is_table', False),
                            'table_id': getattr(chunk, 'table_id', None) if getattr(chunk, 'is_table', False) else None
                        })
                        
                        # Export chunk metadata from Docling
                        if hasattr(chunk, 'meta') and chunk.meta:
                            try:
                                docling_meta = chunk.meta.export_json_dict()
                                chunk_metadata['docling_metadata'] = docling_meta
                            except:
                                pass
                        
                        chunks.append({
                            'text': chunk_text,
                            'chunk_index': i,
                            'total_chunks': len(docling_chunks),
                            'token_count': token_count,
                            'metadata': chunk_metadata,
                            'docling_chunk': chunk  # Keep reference to original chunk
                        })
            else:
                # Fallback to simple chunking if Docling chunking is disabled
                content = self._extract_content_from_docling(docling_doc)
                cleaned_content = self._clean_text(content)
                
                if len(cleaned_content) >= self.config.min_text_length:
                    # Count tokens using chunker's aligned tokenizer
                    token_count = self.docling_chunker.tokenizer.count_tokens(cleaned_content)
                    
                    chunks.append({
                        'text': cleaned_content,
                        'chunk_index': 0,
                        'total_chunks': 1,
                        'token_count': token_count,
                        'metadata': metadata.copy()
                    })
            
            self.logger.info(f"Created {len(chunks)} chunks using Docling hybrid chunker")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to create chunks with Docling: {e}")
            # Fallback to simple chunking
            content = self._extract_content_from_docling(docling_doc)
            cleaned_content = self._clean_text(content)
            
            if len(cleaned_content) >= self.config.min_text_length:
                # Count tokens using chunker's aligned tokenizer
                token_count = self.docling_chunker.tokenizer.count_tokens(cleaned_content)
                
                return [{
                    'text': cleaned_content,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'token_count': token_count,
                    'metadata': metadata.copy()
                }]
            
            return []
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess quality of extracted content"""
        if not content or len(content) < self.config.min_text_length:
            return 0.0
        
        score = 1.0
        
        # Length bonus
        if len(content) >= 100:
            score += 0.1
        elif len(content) < 10:
            score -= 0.2
        
        # Content diversity
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        if total_words > 0:
            diversity_ratio = unique_words / total_words
            if diversity_ratio > 0.7:
                score += 0.1
            elif diversity_ratio < 0.3:
                score -= 0.1
        
        # Punctuation balance
        punct_ratio = len(re.findall(r'[^\w\s]', content)) / len(content)
        if punct_ratio > 0.3:
            score -= 0.1
        elif 0.05 <= punct_ratio <= 0.2:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def process_csv_dataset(self, csv_path: Union[str, Path], output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Process CSV dataset with enhanced text cleaning using Docling principles
        
        Args:
            csv_path: Path to CSV file
            output_path: Optional output path for cleaned CSV
            
        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info(f"🔄 Processing CSV dataset: {Path(csv_path).name}")
            
            # Load CSV
            df = pd.read_csv(csv_path, encoding='utf-8')
            original_count = len(df)
            
            # Clean text columns using Docling-style processing
            df = self._clean_dataframe_text(df)
            
            # Remove low-quality rows
            df = self._remove_low_quality_rows(df)
            
            # Quality assessment
            quality_scores = self._assess_dataframe_quality(df)
            df['quality_score'] = quality_scores
            
            # Final filtering
            df = df[df['quality_score'] >= self.config.quality_threshold].copy()
            
            # Save cleaned data
            if output_path is None:
                output_path = Path(csv_path).parent / f"docling_cleaned_{Path(csv_path).name}"
            
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            self.logger.info(f"✅ CSV dataset processed:")
            self.logger.info(f"   Original rows: {original_count}")
            self.logger.info(f"   Cleaned rows: {len(df)}")
            self.logger.info(f"   Retention: {(len(df)/original_count*100):.1f}%")
            self.logger.info(f"   Output: {output_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process CSV dataset: {e}")
            raise
    
    def batch_process_documents(self, input_dir: Union[str, Path], output_dir: Optional[Path] = None) -> List[ProcessedDocument]:
        """
        Process multiple documents in a directory
        
        Args:
            input_dir: Directory containing documents to process
            output_dir: Optional directory to save processed results
            
        Returns:
            List of ProcessedDocument objects
        """
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_docs = []
        
        try:
            # Find all supported files
            supported_files = []
            for fmt in self.config.supported_formats:
                supported_files.extend(input_dir.glob(f"*.{fmt}"))
                supported_files.extend(input_dir.glob(f"*.{fmt.upper()}"))
            
            self.logger.info(f"🔄 Found {len(supported_files)} supported files to process")
            
            for file_path in supported_files:
                try:
                    # Process document
                    processed_doc = self.process_document(file_path)
                    processed_docs.append(processed_doc)
                    
                    # Save results if output directory specified
                    if output_dir:
                        self._save_processed_document(processed_doc, output_dir)
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to process {file_path.name}: {e}")
                    continue
            
            self.logger.info(f"✅ Batch processing completed: {len(processed_docs)}/{len(supported_files)} documents processed")
            return processed_docs
            
        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            raise
    
    def _clean_dataframe_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns in DataFrame"""
        df_clean = df.copy()
        
        # Get object (string) columns
        object_columns = df_clean.select_dtypes(include=['object']).columns
        
        for col in object_columns:
            df_clean[col] = df_clean[col].astype(str).apply(self._clean_text)
        
        return df_clean
    
    def _remove_low_quality_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with low-quality content"""
        text_columns = df.select_dtypes(include=['object']).columns
        valid_text_counts = df[text_columns].apply(lambda x: x.str.len() > self.config.min_text_length).sum(axis=1)
        
        df_filtered = df[valid_text_counts > 0].copy()
        
        self.logger.info(f"   Removed {len(df) - len(df_filtered)} rows with no valid text")
        return df_filtered
    
    def _assess_dataframe_quality(self, df: pd.DataFrame) -> np.ndarray:
        """Assess quality of each row in DataFrame"""
        quality_scores = []
        
        for _, row in df.iterrows():
            score = self._calculate_row_quality(row)
            quality_scores.append(score)
        
        return np.array(quality_scores)
    
    def _calculate_row_quality(self, row: pd.Series) -> float:
        """Calculate quality score for a single row"""
        score = 1.0
        
        for col, value in row.items():
            if pd.notna(value) and str(value).strip():
                text = str(value).strip()
                
                # Length bonus
                if len(text) >= 10:
                    score += 0.1
                elif len(text) < 3:
                    score -= 0.2
                
                # Content quality indicators
                if text.isupper() and len(text) > 5:
                    score += 0.05  # UI text bonus
                
                if re.search(r'[A-Z]{3,}', text):
                    score += 0.03  # Acronyms bonus
                
                # Penalties
                punct_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
                if punct_ratio > 0.3:
                    score -= 0.1
                
                if re.search(r'(.)\1{2,}', text):
                    score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def _save_processed_document(self, processed_doc: ProcessedDocument, output_dir: Path):
        """Save processed document results"""
        try:
            base_name = Path(processed_doc.file_path).stem
            
            # Save content as text
            content_file = output_dir / f"{base_name}_content.txt"
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(processed_doc.content)
            
            # Save metadata as JSON
            metadata_file = output_dir / f"{base_name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(processed_doc.metadata, f, indent=2, ensure_ascii=False)
            
            # Save chunks as JSON
            chunks_file = output_dir / f"{base_name}_chunks.json"
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(processed_doc.chunks, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"   Saved processed results to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processed document: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing service statistics"""
        return {
            "service": "docling_data_preprocessing",
            "config": {
                "enable_ocr": self.config.enable_ocr,
                "enable_table_extraction": self.config.enable_table_extraction,
                "enable_figure_extraction": self.config.enable_figure_extraction,
                "enable_formula_extraction": self.config.enable_formula_extraction,
                "supported_formats": self.config.supported_formats,
                "min_text_length": self.config.min_text_length,
                "quality_threshold": self.config.quality_threshold,
                "max_chunk_tokens": self.config.max_chunk_tokens
            },
            "docling_converter": "initialized" if self.docling_converter else "not_initialized",
            "docling_chunker": "initialized" if self.docling_chunker else "not_initialized",
            "tokenizer_aligned": "yes" if self.docling_chunker and hasattr(self.docling_chunker, 'tokenizer') else "no",
            "embedding_model": get_preprocessing_config().embedding_model,
            "tokenizer_model": f"{get_preprocessing_config().tokenizer_model} (inferred)"
        }
    
    def prepare_for_qdrant(self, processed_doc: ProcessedDocument) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, float]]]:
        """
        Prepare processed document for Qdrant storage with hybrid vectors
        Returns documents, metadata, and sparse vectors for hybrid search
        
        Based on Docling Qdrant example: https://docling-project.github.io/docling/examples/retrieval_qdrant/
        """
        documents = []
        metadatas = []
        sparse_vectors = []
        
        try:
            for chunk in processed_doc.chunks:
                # Extract text content
                documents.append(chunk['text'])
                
                # Generate sparse vector for hybrid search
                sparse_vector = self._generate_sparse_vector(chunk['text'])
                sparse_vectors.append(sparse_vector)
                
                # Prepare metadata for Qdrant
                qdrant_metadata = chunk['metadata'].copy()
                
                # Add chunk-specific metadata - no domain/language extraction here
                qdrant_metadata.update({
                    'chunk_index': chunk.get('chunk_index', 0),
                    'total_chunks': chunk.get('total_chunks', len(processed_doc.chunks)),
                    'token_count': chunk['token_count'],
                    'chunk_type': chunk.get('chunk_type', 'text'),
                    'is_table': chunk.get('is_table', False),
                    'table_id': chunk.get('table_id'),
                    'file_path': processed_doc.file_path,
                    'file_type': processed_doc.file_type,
                    'quality_score': processed_doc.quality_score,
                    'processing_time': processed_doc.processing_time
                    # Domain, language, source_language, target_language will be provided explicitly during data insertion
                })
                
                # Add Docling metadata if available
                if 'docling_metadata' in chunk:
                    qdrant_metadata['docling_metadata'] = chunk['docling_metadata']
                
                metadatas.append(qdrant_metadata)
            
            self.logger.info(f"Prepared {len(documents)} chunks for Qdrant hybrid storage")
            self.logger.info(f"   Sparse vectors generated: {len(sparse_vectors)}")
            return documents, metadatas, sparse_vectors
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data for Qdrant: {e}")
            return [], [], []
    
    
    
    def batch_prepare_for_qdrant(self, processed_docs: List[ProcessedDocument]) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, float]]]:
        """
        Prepare multiple processed documents for Qdrant storage with hybrid vectors
        """
        all_documents = []
        all_metadatas = []
        all_sparse_vectors = []
        
        for processed_doc in processed_docs:
            documents, metadatas, sparse_vectors = self.prepare_for_qdrant(processed_doc)
            all_documents.extend(documents)
            all_metadatas.extend(metadatas)
            all_sparse_vectors.extend(sparse_vectors)
        
        self.logger.info(f"Batch prepared {len(all_documents)} total chunks for Qdrant hybrid storage")
        return all_documents, all_metadatas, all_sparse_vectors

def create_docling_preprocessing_service(config: Optional[ProcessingConfig] = None) -> DoclingDataPreprocessingService:
    """Factory function to create Docling preprocessing service"""
    return DoclingDataPreprocessingService(config)
