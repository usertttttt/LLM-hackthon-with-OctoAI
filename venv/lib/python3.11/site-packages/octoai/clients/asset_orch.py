"""Asset Orchestrator class."""
from __future__ import (
    annotations,  # required to allow 3.7+ python use type | syntax introduced in 3.10
)

import logging
import os
import re
import time
from abc import ABC
from pathlib import Path
from typing import List, Union

import boto3
import httpx
import yaml

import clients.asset_lake.models as asset_models
import octoai  # for version header
from clients.asset_lake import ApiClient, AssetsApi, Configuration
from clients.asset_lake.models import Status  # used in docstring
from clients.asset_lake.models import (
    AssetType,
    BaseEngine,
    CreateAssetRequest,
    DataType,
    DeleteAssetResponse,
    FileExtension,
    FileFormat,
    PresignedUrlTransferApi,
    StsTransferApi,
    TransferApiType,
)
from octoai.errors import (
    OctoAIAssetReadyTimeoutError,
    OctoAIServerError,
    OctoAIValidationError,
)

LOG = logging.getLogger(__name__)


class Asset:
    """Asset Orchestrator implementation of an asset.

    Hashable with UUID used as key if passed to a dictionary.

    :param id: UUID unique to asset.
    :param asset_type: :class:`AssetType`, including "lora", "vae", "checkpoint", or
        "textual-inversion".
    :param name: Alphanumeric, _, or - allowed.
    :param description: Description of asset.
    :param size_bytes: Total bytes of the asset file.
    :param status: :class:`Status` of Asset.  One of "ready_to_upload", "ready",
        "uploaded", "deleted", "rejected", or "error".
    :param status_details: Description of asset status.
    :param created_at: Time created.
    :param data: Additional information about asset such as engine, file_format, etc.
    :param tenant_uuid: UUID of person who created the asset.
    """

    def __init__(
        self,
        id,
        asset_type,
        name,
        description,
        size_bytes,
        status,
        status_details,
        created_at,
        data,
        tenant_uuid,
        *args,
        **kwargs,  # Catch any vars from al.asset type we don't care about
    ):
        self.id = id
        self.asset_type = asset_type
        self.name = name
        self.description = description
        self.size_bytes = size_bytes
        self.status = status
        self.status_details = status_details
        self.created_at = created_at
        self.data = data
        self.tenant_uuid = tenant_uuid

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"\nid: {self.id}, name: {self.name}, status: {self.status}"

    def __repr__(self):
        return str(self)


class FileData:
    def __init__(self, file_format: FileExtension | str, version: str = ""):
        try:
            file_format = FileExtension(file_format)
        except ValueError:
            msg = (
                f"file_format({file_format}) in `data` field is not valid.  "
                f"Valid options include 'png', 'jpg', 'jpeg', and 'txt'."
            )
            raise OctoAIValidationError(msg)

        self._asset_model = asset_models.FileData(
            asset_type="file", file_format=file_format, version=version
        )

        self.asset_type = AssetType.FILE
        self.file_format = file_format
        self.version = version


class ModelData(ABC):
    """
    Base class for Checkpoints, LoRAs, and Textual Inversions.

    :param engine: Compatible :class:`BaseEngine` type for model.  Includes
        "image/stable-diffusion-v1-5" or "image/stable-diffusion-xl-v1-0".
    :param file_format: :class:`FileFormat` of model, includes `safetensors`.
    :param data_type: :class:`DataType` or str matching an enum in DataType, default to
        'fp16'.
    """

    def __init__(
        self,
        engine: BaseEngine | str,
        file_format: FileFormat | str,
        data_type: DataType | str,
    ):
        try:
            if type(file_format) == str:
                file_format = FileFormat(file_format)
        except ValueError:
            msg = (
                f"file_format({file_format}) in `data` field is not valid.  "
                f"Valid options include 'safetensors'."
            )
            raise OctoAIValidationError(msg)
        if isinstance(file_format, FileFormat):  # Linter handling for unions
            self.file_format = file_format

        try:
            if type(engine) == str:
                engine = BaseEngine(engine)
        except ValueError:
            msg = (
                f"engine({engine}) in `data` field is not valid. Valid options include: "
                ", ".join(BaseEngine.__members__.values())
            )
            raise OctoAIValidationError(msg)
        if isinstance(engine, BaseEngine):
            self.engine = engine

        try:
            if type(data_type) == str:
                data_type = DataType(data_type)
        except ValueError:
            msg = (
                f"data_type({data_type}) in `data` field is not valid.  "
                f"Valid options include 'fp16' and 'fp32'."
            )
            raise OctoAIValidationError(msg)
        if isinstance(data_type, DataType):
            self.data_type = data_type


class CheckpointData(ModelData):
    """
    Checkpoint data associated with `checkpoint` AssetType.

    Used for :meth:`AssetOrchestrator.create`.

    :param engine: Compatible :class:`BaseEngine` type for model.  Includes
        "image/stable-diffusion-v1-5", "image/stable-diffusion-xl-v1-0",
        and "image/controlnet-sdxl".
    :param file_format: :class:`FileFormat` of model, includes `safetensors`.
    :param data_type: :class:`DataType` or str matching an enum in DataType, default to
        'fp16'.
    """

    def __init__(
        self,
        engine: BaseEngine | str,
        file_format: FileFormat | str,
        data_type: DataType | str = DataType.FP16,
    ):
        # Validates and sets params
        super(CheckpointData, self).__init__(engine, file_format, data_type)
        self._asset_model = asset_models.CheckpointData(
            data_type=self.data_type,
            engine=self.engine,
            file_format=self.file_format,
            asset_type=AssetType.CHECKPOINT,
        )
        self.asset_type = AssetType.CHECKPOINT


class LoraData(ModelData):
    """LoRA data associated with the `lora` AssetType.

    Used for :meth:`AssetOrchestrator.create`.

    :param engine: Compatible :class:`BaseEngine` type for model.  Includes
        "image/stable-diffusion-v1-5", "image/stable-diffusion-xl-v1-0",
        and "image/controlnet-sdxl".
    :param file_format: :class:`FileFormat` of model, includes `safetensors`.
    :param data_type: :class:`DataType` or str matching an enum in DataType, default to
        'fp16'.
    """

    def __init__(
        self,
        engine: BaseEngine | str,
        file_format: FileFormat | str,
        data_type: DataType | str = DataType.FP16,
    ):
        super(LoraData, self).__init__(engine, file_format, data_type)
        self._asset_model = asset_models.LoraData(
            data_type=self.data_type,
            engine=self.engine,
            file_format=self.file_format,
            asset_type=AssetType.LORA,
        )
        self.asset_type = AssetType.LORA


class VAEData(ModelData):
    """VAE data associated with the `vae` AssetType.

    Used for :meth:`AssetOrchestrator.create`.

    :param engine: Compatible :class:`BaseEngine` type for model.  Includes
        "image/stable-diffusion-v1-5", "image/stable-diffusion-xl-v1-0",
        and "image/controlnet-sdxl".
    :param file_format: :class:`FileFormat` of model, includes `safetensors`.
    :param data_type: :class:`DataType` or str matching an enum in DataType, default to
        'fp16'.
    """

    def __init__(
        self,
        engine: BaseEngine | str,
        file_format: FileFormat | str,
        data_type: DataType | str = DataType.FP16,
    ):
        super(VAEData, self).__init__(engine, file_format, data_type)
        self._asset_model = asset_models.VAEData(
            data_type=self.data_type,
            engine=self.engine,
            file_format=self.file_format,
            asset_type=AssetType.VAE,
        )
        self.asset_type = AssetType.VAE


class TextualInversionData(ModelData):
    """TextualInversionData associated with `textual_inversion` :class:`AssetType`.

    Used for :meth:`AssetOrchestrator.create`.

    :param engine: Compatible :class:`BaseEngine` type for model.  Includes
        "image/stable-diffusion-v1-5", "image/stable-diffusion-xl-v1-0",
        and "image/controlnet-sdxl".
    :param file_format: :class:`FileFormat` of model, includes `safetensors`.
    :param data_type: :class:`DataType` or str matching an enum in DataType, default to
        'fp16'.
    """

    def __init__(
        self,
        engine: BaseEngine | str,
        file_format: FileFormat | str,
        trigger_words: List[str],
        data_type: DataType | str = DataType.FP16,
    ):
        super(TextualInversionData, self).__init__(engine, file_format, data_type)
        self._asset_model = asset_models.TextualInversionData(
            data_type=self.data_type,
            engine=self.engine,
            file_format=self.file_format,
            asset_type=AssetType.TEXTUAL_INVERSION,
            trigger_words=trigger_words,
        )
        self.asset_type = AssetType.TEXTUAL_INVERSION
        self.trigger_words = trigger_words


AssetData = Union[
    FileData,
    CheckpointData,
    LoraData,
    TextualInversionData,
    VAEData,
]


class AssetOrchestrator:
    """Asset Orchestrator class to create, read, delete, and list assets.

    :param token: OCTOAI_TOKEN if one is not set as an environment variable, default to
        None.
    :param config_path: Path to config file from CLI, used if token or envvar is not
        set, default to None and checks default path.
    :param endpoint: Defaults to "https://api.octoai.cloud/".
    """

    def __init__(
        self,
        token: str | None = None,
        config_path: str | None = None,
        endpoint: str = "https://api.octoai.cloud/",
    ):
        # Would benefit from refactoring this with client code
        if token is None:
            token = token if token else os.environ.get("OCTOAI_TOKEN", None)

        if not token:
            # Default path is ~/.octoai/config.yaml for token, can be overridden
            path = Path(config_path) if config_path else Path.home()
            try:
                with open(
                    (path / Path(".octoai/config.yaml")), encoding="utf-8"
                ) as octoai_config_yaml:
                    config_dict = yaml.safe_load(octoai_config_yaml)
                token = config_dict.get("token")
            except FileNotFoundError:
                token = None

        if not token:
            msg = (
                "Authorization is required.  Please set an `OCTOAI_TOKEN` "
                "environment variable, or pass your token to the client using "
                "`asset_client = AssetOrchestrator(token='your-octoai-api-token')`"
            )
            raise OctoAIValidationError(msg)

        conf = Configuration(host=endpoint)
        temp_client = ApiClient(
            conf, header_name="Authorization", header_value=f"Bearer {token}"
        )
        version = octoai.__version__
        temp_client.user_agent = f"octoai-{version}"
        self._auth_header = f"Bearer {token}"
        self.client = AssetsApi(api_client=temp_client)

    def create(
        self,
        data: AssetData,
        name: str,  # only alphanumeric characters, _, -
        file: str | None = None,
        url: str | None = None,
        is_public: bool = False,
        description: str | None = None,
        transfer_api_type: str | TransferApiType | None = None,
    ) -> Asset:
        """
        Create and upload an asset.

        :param data: :class:`CheckpointData`, :class:`LoraData`, :class:`VAEData`,
            or :class:`TextualInversionData`
        :param name: Name of asset, alphanumeric with `-` and `_` characters allowed.
        :param file: str to file path, optional, defaults to `None`.
        :param url: Url to copy file data from instead of file, optional, defaults to `None`. If set `file` and `transfer_api_type` must be None.
        :param description: Description of asset, optional, defaults to `None`.
        :param transfer_api_type: :class:`TransferApiType` or str of either
            "presigned-url", or "sts", defaults to "sts" for >= 50mb and
            "presigned-url" for under 50mb.
        :return: :class:`Asset`
        """
        self._validate_name(name)
        if transfer_api_type is None and url is None:  # Pick best option for file size
            if os.path.getsize(file) >= 52428800:
                transfer_api_type = TransferApiType.STS
            else:
                transfer_api_type = TransferApiType.PRESIGNED_MINUS_URL
        if isinstance(transfer_api_type, str):  # Appease the linter
            transfer_api_type = TransferApiType(transfer_api_type)
        request = CreateAssetRequest(
            name=name,
            description=description,
            url=url,
            asset_type=data.asset_type,
            is_public=is_public,
            data=data._asset_model.to_dict(),
            transfer_api_type=transfer_api_type,
        )

        response = self.client.create_asset_v1_assets_post(request)

        if transfer_api_type is not None:
            x_api = response.transfer_api.actual_instance
            if isinstance(x_api, PresignedUrlTransferApi):
                with open(file, "rb") as file_data:
                    upload_resp = httpx.put(
                        url=x_api.put_url, content=file_data, timeout=60000
                    )
            if isinstance(x_api, StsTransferApi):
                self._sts_upload(x_api, file)

            # Get asset after upload to verify correct status
            self.client.complete_asset_upload_v1_assets_asset_id_complete_upload_post(
                response.asset.id,
                asset_models.CompleteAssetUploadRequest(token=response.token),
            )

        asset = self.get(response.asset.id)
        return asset

    def wait_for_ready(
        self, asset: Asset, poll_interval=10, timeout_seconds=900
    ) -> Asset:
        """
        Wait for asset to be ready to use.

        This waits until the asset's status is READY or an error status.

        :param asset: Asset to wait on
        """
        timer = 0
        while (
            asset.status != Status.READY
            and asset.status != Status.ERROR
            and asset.status != Status.REJECTED
        ):
            time.sleep(poll_interval)
            timer += poll_interval
            if timer > timeout_seconds:
                raise OctoAIAssetReadyTimeoutError(
                    f"Asset {asset.id} was not ready after {timeout_seconds} seconds"
                )
            asset = self.get(id=asset.id)

        if asset.status != Status.READY:
            raise OctoAIValidationError(
                f"Error validating asset {asset.id}: {asset.status} {asset.status_details}."
            )

        return asset

    @staticmethod
    def _sts_upload(x_api: StsTransferApi, file: str):
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=x_api.aws_access_key_id,
            aws_secret_access_key=x_api.aws_secret_access_key,
            aws_session_token=x_api.aws_session_token,
        )
        try:
            s3_client.upload_file(file, x_api.s3_bucket, x_api.s3_key)
        except Exception as e:
            raise OctoAIServerError(f"Error uploading file to server: {e}")

    def list(
        self,
        name: str | None = None,
        is_public: bool | None = None,
        data_type: DataType | None = None,
        asset_type: List[AssetType] | dict | None = None,
        engine: List[BaseEngine] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        owner: str | None = None,
    ) -> list[Asset]:
        """
        Return list of assets filtered on the non-None parameters.

        :param name: Asset name, alphanumeric, -, and _ allowed.  Defaults to None.
        :param is_public: Whether to filter for public assets, such as looking for
            `octoai` public assets.
        :param data_type: :class:`DataType`, defaults to None.
        :param asset_type: List of :class:`AssetType` of assets, defaults to None.
        :param engine: List of :class:`BaseEngine` of assets, defaults to None.
        :param limit: Max number of assets to return, defaults to None.
        :param offset: Where to start including next list of assets, defaults to None.
        :param owner: ID of owner, defaults to None.
        :return: list[:class:`Asset`]
        """
        if name is not None:
            self._validate_name(name)
        self._validate_limit(limit)
        self._validate_offset(offset)
        inputs = locals()
        inputs.pop("self")
        for key in list(inputs):
            if inputs[key] is None:
                inputs.pop(key)
        response = self.client.list_assets_v1_assets_get(**inputs)
        result = []
        for item in response.data:
            result.append(Asset(**item.to_dict()))
        return result

    def get(self, name: str | None = None, id: str | None = None) -> Asset:
        """
        Get an asset associated with an asset name or asset id.

        :param name: Name of the asset to get.
        :param id: ID of the asset to get.
        :return: :class:`Asset`
        """
        if name is None and id is None:
            raise OctoAIValidationError("Either name or id must be provided.")
        if name is not None and id is not None:
            raise OctoAIValidationError("Only one of name or id may be provided.")

        asset_name_or_id = name if name is not None else id
        self._validate_name(asset_name_or_id)
        response = self.client.retrieve_asset_v1_assets_asset_owner_and_name_or_id_get(
            asset_name_or_id
        )
        asset = Asset(**response.asset.to_dict())
        return asset

    def delete(self, asset_id: str | None) -> DeleteAssetResponse:
        """Delete an asset.

        :param asset_id: the UUID of the asset to be deleted.
        :return: :class:`DeleteAssetResponse` containing an `id` as a str and
            `deleted_at` as a str with the timestamp.
        """
        response = self.client.delete_asset_v1_assets_asset_id_delete(asset_id)
        return response

    @staticmethod
    def _validate_name(name: str):
        if re.fullmatch("^[a-zA-Z0-9_-]*$", name) is None:
            msg = (
                f"name or id({name}) is invalid.  Valid names or ids may only contain "
                f"alphanumeric, '-', or '_' characters."
            )
            raise OctoAIValidationError(msg)

    @staticmethod
    def _validate_offset(offset: int | None):
        if offset is not None and (type(offset) is not int or offset < 0):
            msg = (
                f"offset({offset}) is invalid. Valid offsets must be ints >= 0 or None."
            )
            raise OctoAIValidationError(msg)

    @staticmethod
    def _validate_limit(limit: int):
        if limit is not None and (type(limit) is not int or limit <= 0 or limit > 100):
            msg = (
                f"limit({limit}) is invalid. Valid limits are ints > 0 and <= 100 or "
                f"None."
            )
            raise OctoAIValidationError(msg)
