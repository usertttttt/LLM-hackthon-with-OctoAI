"""Example OctoAI service scaffold: Hello World."""
from octoai.service import Service


class HelloService(Service):
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
        """Perform intialization."""
        print("Setting up.")

    def infer(self, prompt: str) -> str:
        """Perform inference."""
        return prompt
