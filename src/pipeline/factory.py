"""Pipeline factory — builds RAG components from a PipelineConfig.

Each method returns an instance of the corresponding interface.
Swapping strategies = changing the config → factory returns a different implementation.
"""

from src.config import PipelineConfig
from src.pipeline import Chunker, Embedder, Generator, Parser, QueryRewriter, Reranker, Retriever
from src.pipeline.parser import PyMuPDFParser


def build_parser(config: PipelineConfig) -> Parser:
    return PyMuPDFParser()


def build_chunker(config: PipelineConfig) -> Chunker:
    raise NotImplementedError


def build_embedder(config: PipelineConfig) -> Embedder:
    raise NotImplementedError


def build_retriever(config: PipelineConfig) -> Retriever:
    raise NotImplementedError


def build_reranker(config: PipelineConfig) -> Reranker:
    raise NotImplementedError


def build_rewriter(config: PipelineConfig) -> QueryRewriter:
    raise NotImplementedError


def build_generator(config: PipelineConfig) -> Generator:
    raise NotImplementedError
