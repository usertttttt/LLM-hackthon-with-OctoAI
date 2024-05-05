"""Example OctoAI service scaffold: Flan-T5-Small."""
from transformers import T5ForConditionalGeneration, T5Tokenizer

from octoai.service import Service

"""
Flan-T5 is an instruction-finetuned version of T5, a text-to-text
transformer language model.
"""


class T5Service(Service):
    """An OctoAI service extends octoai.service.Service."""

    # Uncomment this function if you want to place large assets
    # (checkpoints, models, etc) on a docker volume instead of building
    # them into the image. This can provide faster
    # cold-start time. You may need to modify your inference pipeline
    # so that it leverages the locally-stored assets instead of
    # downloading them at container boot time.
    # def store_assets(self) -> None:
    #     """Download model assets."""
    #     # By default, instantiate the model to fill local disk
    #     # caches on the OctoAI volume. You can customize this if
    #     # your model can be fetched without instantiating it.
    #     self.setup()

    def setup(self):
        """Download model weights to disk."""
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    def infer(self, prompt: str) -> str:
        """Perform inference with the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return response[0]
