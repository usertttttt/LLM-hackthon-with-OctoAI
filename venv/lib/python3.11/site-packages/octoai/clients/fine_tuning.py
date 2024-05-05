"""OctoAI Fine Tuning."""
from __future__ import (
    annotations,  # required to allow 3.7+ python use type | syntax introduced in 3.10
)

import re
from typing import List, Mapping

import octoai  # for version header
from clients.fine_tuning import ApiClient, Configuration, TuneApi
from clients.fine_tuning.models import (
    BaseEngine,
    CreateTuneRequest,
    Details,
    ListTunesResponse,
    LoraTuneCheckpoint,
    LoraTuneFile,
    LoraTuneInput,
    Tune,
)
from octoai.clients.asset_orch import Asset
from octoai.errors import OctoAIValidationError


class FineTuningClient:
    """
    OctoAI Fine Tuning Client.

    This client is used to interact with the OctoAI Fine Tuning API.
    This class can be used on its own, providing a valid token and endpoint,
    but it is mainly used from instances of :class: `octoai.client.Client` as
    `client.tune`.
    """

    def __init__(
        self,
        token: str,
        endpoint: str = "https://api.octoai.cloud/",
    ):
        if token is None:
            raise OctoAIValidationError(
                "Authorization is required for fine tuning. "
                "Please provide a valid OctoAI token."
            )

        self.token = token
        conf = Configuration(host=endpoint)
        temp_client = ApiClient(
            conf, header_name="Authorization", header_value=f"Bearer {token}"
        )
        version = octoai.__version__
        temp_client.user_agent = f"octoai-{version}"
        self._auth_header = f"Bearer {token}"
        self.client = TuneApi(api_client=temp_client)

    @staticmethod
    def _convert_files_to_lora_tune_files(
        files: List[LoraTuneFile]
        | List[Asset]
        | List[str]
        | Mapping[Asset, str]
        | Mapping[str, str]
    ) -> List[LoraTuneFile]:
        if isinstance(files, list):
            if isinstance(files[0], Asset):
                return [LoraTuneFile(file_id=file.id) for file in files]
            elif isinstance(files[0], str):
                return [LoraTuneFile(file_id=file) for file in files]
            elif isinstance(files[0], LoraTuneFile):
                return files
        if isinstance(files, Mapping):
            if isinstance(next(iter(files.keys())), Asset):
                return [
                    LoraTuneFile(file_id=file.id, caption=caption)
                    for file, caption in files.items()
                ]
            elif isinstance(next(iter(files.keys())), str):
                return [
                    LoraTuneFile(file_id=file, caption=caption)
                    for file, caption in files.items()
                ]

    def create(
        self,
        name: str,
        base_checkpoint: str | Asset,
        files: List[LoraTuneFile]
        | List[Asset]
        | List[str]
        | Mapping[Asset, str]
        | Mapping[str, str],
        trigger_words: str | List[str],
        steps: int,
        description: str | None = None,
        engine: str | BaseEngine | None = None,
        seed: int | None = None,
        continue_on_rejection: bool = False,
    ) -> Tune:
        """
        Create a new fine tuning job.

        :param name: Required. The name of the fine tuning job.
        :param base_checkpoint: Required. The base checkpoint to use.
            Accepts an asset id or an asset object.
        :param files: Required. The training files to use.
            Supports a list of assets or asset ids without captions, or a mapping of
            assets or asset ids to captions.
        :param trigger_words: Required. The trigger words to use.
        :steps: Required. The number of steps to train for.
        :param description: Optional. The description of the fine tuning job.
        :param engine: Optional. The engine to use. Defaults to the corresponding
            engine for the base checkpoint.
        :param seed: Optional. The seed to use for training. Defaults to a random seed.
        :param continue_on_rejection: Optionally continue with the fine-tune job if any
            of the training images are identified as NSFW.  Defaults to False.
        :return: :class:`Tune` object representing the newly created fine tuning job.
        """
        # ensure correct input types
        self._validate_name(name)
        self._validate_files(files)
        self._validate_trigger_words(trigger_words)
        self._validate_steps(steps)
        self._validate_engine(engine)

        # convert to codegen-compatible types
        files = self._convert_files_to_lora_tune_files(files)
        if isinstance(base_checkpoint, Asset):
            base_checkpoint = base_checkpoint.id
        if isinstance(engine, str):
            engine = BaseEngine(engine)
        if isinstance(trigger_words, str):
            trigger_words = [trigger_words]

        lti = LoraTuneInput(
            base_checkpoint=LoraTuneCheckpoint(
                checkpoint_id=base_checkpoint,
                engine=engine,
            ),
            files=files,
            trigger_words=trigger_words,
            steps=steps,
            seed=seed,
            # this is a little strange we need to pass this here
            # but it's required by the codegen model
            tune_type="lora_tune",
        )

        # call endpoint using codegen APIs
        request = CreateTuneRequest(
            name=name,
            description=description,
            details=Details(
                LoraTuneInput(
                    base_checkpoint=LoraTuneCheckpoint(
                        checkpoint_id=base_checkpoint,
                        engine=engine,
                    ),
                    files=files,
                    trigger_words=trigger_words,
                    steps=steps,
                    seed=seed,
                    # this is a little strange we need to pass this here
                    # but it's required by the codegen model
                    tune_type="lora_tune",
                )
            ),
            continue_on_rejection=continue_on_rejection,
        )

        tune = self.client.create_tune_v1_tune_post(request)
        return tune

    def get(self, id: str) -> Tune:
        """
        Get a fine tuning job.

        :param id: Required. The id of the fine tuning job.
        :return: :class:`Tune` object representing the fine tuning job.
        """
        self._validate_id(id)

        return self.client.get_tune_v1_tune_tune_id_get(id)

    def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        name: str | None = None,
        tune_type: str | None = None,
        base_checkpoint: str | Asset | None = None,
        trigger_words: str | List[str] | None = None,
    ) -> ListTunesResponse:
        """
        List available fine tuning jobs.

        :param limit: Optional. The maximum number of fine tuning jobs to return.
        :param offset: Optional. The offset to start listing fine tuning jobs from.
        :param name: Optional. Filter results by job name.
        :param tune_type: Optional. Filter results by job type.
        :param base_checkpoint: Optional. Filter results by base checkpoint.
            Accepts an asset id or an asset object.
        :param trigger_words: Optional. Filter results by trigger words.
        :return: :class:`ListTunesResponse` object representing the list of fine tuning jobs.
        """
        if name:
            self._validate_name(name)
        if trigger_words:
            self._validate_trigger_words(trigger_words)

        if isinstance(base_checkpoint, Asset):
            base_checkpoint = base_checkpoint.asset_id
        if isinstance(trigger_words, str):
            trigger_words = [trigger_words]

        return self.client.list_tunes_v1_tunes_get(
            limit=limit,
            offset=offset,
            name=name,
            tune_type=tune_type,
            base_checkpoint_id=base_checkpoint,
            trigger_words=trigger_words,
        )

    def delete(self, id: str):
        """
        Delete a fine tuning job.

        :param id: Required. The id of the fine tuning job to delete.
        """
        self._validate_id(id)

        self.client.delete_tune_v1_tune_tune_id_delete(id)

    def cancel(self, id: str) -> Tune:
        """
        Cancel a fine tuning job.

        :param id: Required. The id of the fine tuning job to cancel.
        :return: :class:`Tune` object representing the fine tuning job.
        """
        self._validate_id(id)

        return self.client.cancel_tune_v1_tune_tune_id_cancel_post(id)

    @staticmethod
    def _validate_id(id: str):
        if id is None or len(id) == 0:
            raise OctoAIValidationError("id is required.")

    @staticmethod
    def _validate_name(name: str):
        if re.fullmatch("^[a-zA-Z0-9_-]*$", name) is None:
            msg = (
                f"name({name}) is invalid.  Valid names may only contain "
                f"alphanumeric, '-', or '_' characters."
            )
            raise OctoAIValidationError(msg)

    # this is tediuous but the intent is to offer a flexible API and this is the cost
    # to prevent cryptic errors from codegen code or the back end
    @staticmethod
    def _validate_files(
        files: List[LoraTuneFile]
        | List[Asset]
        | List[str]
        | Mapping[Asset, str]
        | Mapping[str, str]
    ):
        if isinstance(files, list):

            def raise_list_error(first: str):
                raise OctoAIValidationError(
                    f"files({files}) is invalid. Valid files may only contain "
                    f"LoraTuneFile, Asset, or str. The first file was of type {first}, and "
                    f"at least one file in the list was not"
                )

            if len(files) == 0:
                raise OctoAIValidationError("files must contain at least one file.")
            elif isinstance(files[0], LoraTuneFile):
                if not all(isinstance(file, LoraTuneFile) for file in files):
                    raise_list_error("LoraTuneFile")
            elif isinstance(files[0], Asset):
                if not all(isinstance(file, Asset) for file in files):
                    raise_list_error("Asset")
            elif isinstance(files[0], str):
                if not all(isinstance(file, str) for file in files):
                    raise_list_error("str")
            else:
                raise OctoAIValidationError(
                    f"files({files}) is invalid. Valid files may only contain "
                    f"LoraTuneFile, Asset, or str (asset ids)."
                )

        if isinstance(files, Mapping):
            if len(files) == 0:
                raise OctoAIValidationError("files must contain at least one file.")
            elif isinstance(next(iter(files.keys())), Asset):
                if not all(isinstance(file, Asset) for file in files.keys()):
                    raise OctoAIValidationError("All files keys must be of type Asset.")
                if not all(isinstance(file, str) for file in files.values()):
                    raise OctoAIValidationError(
                        "All files values must be of type str (captions)."
                    )
            elif isinstance(next(iter(files.keys())), str):
                if not all(isinstance(file, str) for file in files.keys()):
                    raise OctoAIValidationError(
                        "All files keys must be of type str (asset ids)."
                    )
                if not all(isinstance(file, str) for file in files.values()):
                    raise OctoAIValidationError(
                        "All files values must be of type str (captions)."
                    )
            else:
                raise OctoAIValidationError(
                    f"The files({files}) mapping is invalid. Valid files keys may only contain "
                    f"Asset or str (asset ids)."
                )

    @staticmethod
    def _validate_trigger_words(trigger_words: str | List[str]):
        if isinstance(trigger_words, str):
            if len(trigger_words) == 0:
                raise OctoAIValidationError(
                    "trigger_words must contain at least one word."
                )
        if not all(isinstance(trigger_word, str) for trigger_word in trigger_words):
            raise OctoAIValidationError(
                f"trigger_words({trigger_words}) is invalid. Valid trigger_words may only contain "
                f"str."
            )
        if not all(len(trigger_word) > 0 for trigger_word in trigger_words):
            raise OctoAIValidationError(
                f"trigger_words({trigger_words}) is invalid. Valid trigger_words may only contain "
                f"non-empty str."
            )

    @staticmethod
    def _validate_steps(steps: int):
        if steps <= 0:
            raise OctoAIValidationError("steps must be greater than 0.")

    @staticmethod
    def _validate_engine(engine: str | BaseEngine | None = None):
        if engine is None:
            return
        elif isinstance(engine, str):
            if engine not in [e.value for e in BaseEngine]:
                raise OctoAIValidationError(
                    f"engine({engine}) is invalid. Valid engines are: "
                    f"{[e.value for e in BaseEngine]}"
                )
        elif not isinstance(engine, BaseEngine):
            raise OctoAIValidationError("engine must be of type str or BaseEngine.")
