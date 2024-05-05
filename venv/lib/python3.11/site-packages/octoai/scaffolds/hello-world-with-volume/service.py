"""Example OctoAI service scaffold: Hello World."""
import pathlib

from octoai.service import Service, volume_path

ASSET_FILENAME: str = "asset.txt"


class HelloService(Service):
    """An OctoAI service extends octoai.service.Service."""

    def store_assets(self) -> None:
        """Download model assets."""
        root_path = pathlib.Path(volume_path())
        with open(root_path / ASSET_FILENAME, "w") as f:
            f.write("using volume")

    def setup(self):
        """Perform intialization."""
        print("Setting up.")
        root_path = pathlib.Path(volume_path())
        with open(root_path / ASSET_FILENAME, "r") as f:
            self.asset = f.read()

    def infer(self, prompt: str) -> str:
        """Perform inference."""
        return prompt + f" {self.asset}!"
