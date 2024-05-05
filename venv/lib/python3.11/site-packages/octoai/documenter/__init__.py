"""Generate examples for the ``octoai.service.Service`` subclass."""

from enum import Enum

from ..service import Service
from ..spec import parse_service
from .common import DocumentationGenerator, MustacheGenerationResult
from .curl import CurlDocumentationGenerator
from .python import PythonExampleGenerator


class DocumentationFormat(Enum):
    """The format of the documentation generated."""

    CURL = 1
    PYTHON = 2


def generate_usage_examples(
    service: Service, format: DocumentationFormat
) -> MustacheGenerationResult:
    """Generate examples for the specified service."""
    generator: DocumentationGenerator
    if format == DocumentationFormat.CURL:
        generator = CurlDocumentationGenerator()
    elif format == DocumentationFormat.PYTHON:
        generator = PythonExampleGenerator()
    else:
        raise ValueError(f"Invalid documentation format f{format}")

    return generator.generate(parse_service(type(service)))


__all__ = ["generate_usage_examples", "DocumentationFormat", "MustacheGenerationResult"]
