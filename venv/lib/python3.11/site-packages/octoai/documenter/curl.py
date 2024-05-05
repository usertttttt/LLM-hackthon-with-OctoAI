"""Generate cURL examples."""

from octoai.documenter.common import (
    Buffer,
    DocumentationGenerator,
    MustacheGenerationResult,
)
from octoai.documenter.json import JSONGeneratingVisitor
from octoai.spec.ast import Field, InferenceService


class CurlDocumentationGenerator(DocumentationGenerator):
    """Generate cURL documentation."""

    def generate(self, service: InferenceService) -> MustacheGenerationResult:
        """Generate usage example given the parsed ``InferenceService``.

        :param service: the parsed ``InferenceService``.
        """
        generation_template = self._get_generation_template()

        input_result = self._generate_input(service.inference_endpoint.inputs)

        full_result = generation_template.format(
            input_placeholder=input_result.generated_template,
        )

        return MustacheGenerationResult(
            generated_template=full_result,
            template_hash={
                "endpoint": "<endpoint>",
                "access_token": "$OCTOAI_TOKEN",
                **input_result.template_hash,
            },
        )

    def _get_generation_template(self) -> str:
        return """curl {{{{{{endpoint}}}}}}/infer \\
    -X POST \\{{{{#access_token}}}}
    -H "Authorization: Bearer {{{{{{access_token}}}}}}" \\{{{{/access_token}}}}
    -d @- << 'EOF'{input_placeholder}
    EOF
"""

    def _generate_input(self, field: Field) -> MustacheGenerationResult:
        generator = JSONGeneratingVisitor(base_buffer=Buffer(indentation_level=1))
        data = generator.generate(field)
        return data
