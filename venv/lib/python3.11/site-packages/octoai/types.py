"""
Type definitions to help communicate with endpoints.

These type definitions help with routine format conversions
that are necessary when transferring binary files (such as
images or audio) over HTTP. These type definitions can be useful
both when creating endpoints (implementing ``octoai.service.Service``
as directed by the ``octoai`` command-line interface) and when
communicating with live endpoints using the client SDK.
"""

import base64
import importlib
import os
import sys
import tempfile
from io import BytesIO
from types import ModuleType
from typing import Any, Dict, Iterator, List, Tuple, Union, cast

import httpx
from fastapi import UploadFile
from numpy.typing import ArrayLike
from PIL import Image as PImage
from pydantic import BaseModel, Field, HttpUrl, field_serializer


def _import_soundfile() -> ModuleType:
    """Attempt to import the soundfile module."""
    if "soundfile" in sys.modules:
        return sys.modules["soundfile"]

    try:
        return importlib.import_module("soundfile")
    except OSError:
        raise Exception(
            "Can't import the 'soundfile' module. "
            "For Linux, try: sudo apt install libsndfile1. "
            "For Mac, try: brew install libsndfile."
        )


def _import_imageio() -> ModuleType:
    """Attempt to import the imageio.v3 module."""
    if "imageio.v3" in sys.modules:
        return sys.modules["imageio.v3"]

    try:
        return importlib.import_module("imageio.v3")
    except OSError:
        raise Exception(
            "Can't import the 'imageio.v3' module. "
            "Try reinstalling octoai-sdk with the following command: "
            "pip install octoai-sdk[video]"
        )


class MediaUrl(BaseModel):
    """URL reference to a media file."""

    url: HttpUrl

    @field_serializer("url")
    def url2str(self, val) -> str:
        """Serialize HttpUrl to str."""
        return str(val)


class Media(BaseModel):
    """Base class for media files.

    The `Media` class is a wrapper for either binary media content
    or a URL reference to a media file. When an instance contains
    binary media data, it is stored and transferred over HTTP as a
    base64-encoded string. When an instance contains a URL reference,
    the URL is stored and transferred over HTTP as a string. The
    URL representation can be more advantageous when the media file is
    already hosted on a server, as it avoids the overhead of base64
    encoding and decoding.

    The `Media` class is the base class for the `Image`, `Audio`, and
    `Video` classes, which contain additional convenience methods to
    read and write media files in common formats. These subclasses
    should be used instead of the `Media` in most cases.
    """

    data: Union[bytes, MediaUrl] = Field(
        description="URL reference or base64-encoded data"
    )

    def __init__(self, data: Union[bytes, MediaUrl]):
        super().__init__(data=data)

    @classmethod
    def from_base64(cls, b64: bytes) -> "Media":
        """Create from base64-encoded data, such as that returned from an HTTP call.

        See also ``Media.from_endpoint_response()``.

        :param b64: base64-encoded data representing a media file
        :type b64: bytes
        :return: ``Media`` instance
        :rtype: :class:`Media`
        """
        return cls(data=b64)

    @classmethod
    def from_endpoint_response(cls, resp_dict: Dict[str, Any], key: str) -> "Media":
        """Create from an endpoint response, such as an endpoint that produces media.

        :param resp_dict: a response from an OctoAI endpoint that produces media
        :type resp_dict: Dict[str, Any]
        :param key: the key name in the response that contains the media
        :return: `Media` object
        :rtype: :class:`Media`
        """
        if key in resp_dict:
            return cls(**resp_dict[key])
        elif "output" in resp_dict and key in resp_dict["output"]:
            return cls(**resp_dict["output"][key])

        raise ValueError(f"Key {key} not found in response")

    @classmethod
    def from_file(cls, file_name: str) -> "Media":
        """Create media from local file.

        :param file_name: path to local media file
        :type file_name: str
        :raises ValueError: file_name not found at provided path
        :return: `Media` object
        :rtype: :class:`Media`
        """
        if not os.path.isfile(file_name):
            raise ValueError(f"File {file_name} does not exist")

        with open(file_name, "rb") as fd:
            return cls(data=base64.b64encode(fd.read()))

    @classmethod
    def from_url(cls, url: str, b64=False, follow_redirects=False) -> "Media":
        """Create media from URL.

        :param url: URL to media file
        :type url: str
        :param b64: whether to base64-encode the media file, defaults to False
        :type b64: bool, optional
        :param follow_redirects: whether to follow redirects, defaults to False
        :type follow_redirects: bool, optional
        :rtype: :class:`Media`
        """
        if b64:
            resp = httpx.get(url, follow_redirects=follow_redirects)
            if resp.status_code != 200:
                raise ValueError(f"status {resp.status_code} ({url})")

            return cls(data=base64.b64encode(resp.content))
        else:
            return cls(data=MediaUrl(url=HttpUrl(url)))

    def download(self, follow_redirects, file_name):
        """Download media URL to local file.

        :param follow_redirects: whether to follow redirects
        :type follow_redirects: bool
        :param file_name: path to local file
        :type file_name: str
        """
        if isinstance(self.data, bytes):
            return
        elif isinstance(self.data, MediaUrl):
            with httpx.stream(
                "GET",
                url=str(self.data.url),
                follow_redirects=follow_redirects,
            ) as resp:
                if resp.status_code != 200:
                    raise ValueError(
                        f"status {resp.status_code} ({str(self.data.url)})"
                    )
                for chunk in resp.iter_bytes():
                    with open(file_name, "ab") as f:
                        f.write(chunk)

    def default_download(self, file_name: str):
        """Download media URL to a file using the default settings.

        Override this to change the download behavior.

        :param file_name: path to local file
        :type file_name: str
        """
        self.download(follow_redirects=False, file_name=file_name)

    def to_file(self, file_name: str):
        """Write media to local file.

        :param file_name: path to local file
        :type file_name: str
        """
        if isinstance(self.data, bytes):
            with open(file_name, "wb") as fd:
                fd.write(base64.b64decode(self.data))
        elif isinstance(self.data, MediaUrl):
            self.default_download(file_name=file_name)

    def to_bytes(self) -> bytes:
        """Convert media to bytes.

        :return: media as bytes
        :rtype: bytes
        """
        if isinstance(self.data, bytes):
            return base64.b64decode(self.data)

        if isinstance(self.data, MediaUrl):
            with tempfile.NamedTemporaryFile() as f:
                self.default_download(file_name=f.name)
                with open(f.name, "rb") as fd:
                    return fd.read()

    def to_base64(self) -> bytes:
        """Convert media to base64-encoded bytes.

        :return: media as base64-encoded bytes
        :rtype: bytes
        """
        if isinstance(self.data, bytes):
            return self.data

        if isinstance(self.data, MediaUrl):
            with tempfile.NamedTemporaryFile() as f:
                self.default_download(file_name=f.name)
                with open(f.name, "rb") as fd:
                    return base64.b64encode(fd.read())


class Image(Media):
    """Image helpers for models that accept or return images.

    The `Image` class is a wrapper for either binary image content
    or a URL reference to an image file. When an instance contains
    binary image data, it is stored and transferred over HTTP as a
    base64-encoded string. When an instance contains a URL reference,
    the URL is stored and transferred over HTTP as a string. The
    URL representation can be more advantageous when the image file is
    already hosted on a server, as it avoids the overhead of base64
    encoding and decoding.

    The `Image` class contains additional convenience methods to
    read and write image files in common formats.
    """

    @classmethod
    def from_pil(cls, image_pil: PImage, format="JPEG") -> "Image":
        """Create from Pillow image object.

        A file format is required since the Pillow image object is
        serialized to a binary image file. The default is "JPEG".

        :param image_pil: image in PIL format
        :type image_pil: PIL.Image
        :param format: target file format, defaults to "JPEG"
        :type format: str, optional
        :return: `Image` object
        :rtype: :class:`Image`
        """
        buffer = BytesIO()
        image_pil.save(buffer, format=format)
        return cls(data=base64.b64encode(buffer.getvalue()))

    def to_pil(self) -> PImage:
        """Convert to PIL Image.

        :return: Pillow image object
        :rtype: PIL.Image
        """
        return PImage.open(BytesIO(self.to_bytes()))

    def is_valid(self):
        """Check if this is a valid image.

        :return: True if valid, False if invalid
        :rtype: bool
        """
        try:
            self.to_pil().verify()
            return True
        except Exception:
            return False


class Audio(Media):
    """Audio helpers for models that accept or return audio.

    The `Audio` class is a wrapper for either binary audio content
    or a URL reference to an audio file. When an instance contains
    binary audio data, it is stored and transferred over HTTP as a
    base64-encoded string. When an instance contains a URL reference,
    the URL is stored and transferred over HTTP as a string. The
    URL representation can be more advantageous when the audio file is
    already hosted on a server, as it avoids the overhead of base64
    encoding and decoding.

    The `Audio` class contains additional convenience methods to
    read and write audio files in common formats.
    """

    @classmethod
    def from_numpy(cls, data: ArrayLike, sample_rate: int, format="WAV") -> "Audio":
        """Create from a numpy array.

        The first dimension of the array should represent audio frames (samples),
        while the second dimension should represent audio channels.

        A file format and a sample rate are needed since the audio data is
        serialized to a binary audio file. The default format is "WAV", and the
        sample rate is required.

        :param data: numpy array with audio data (frames x channels)
        :type data: ArrayLike
        :param sample_rate: samples per second taken to create signal
        :type sample_rate: int
        :param format: target format, defaults to "WAV"
        :type format: str, optional
        :return: Audio object
        :rtype: :class:`Audio`
        """
        soundfile = _import_soundfile()
        buffer = BytesIO()
        soundfile.write(buffer, data=data, samplerate=sample_rate, format=format)
        return cls(data=base64.b64encode(buffer.getvalue()))

    def to_numpy(self) -> Tuple[ArrayLike, int]:
        """Convert to numpy array.

        :return: numpy array representation (frames x channels)
        :rtype: Tuple[ArrayLike, int]
        """
        soundfile = _import_soundfile()
        fd = BytesIO(self.to_bytes())
        data, sample_rate = soundfile.read(fd)
        return (data, sample_rate)

    def is_valid(self):
        """Check if this is a valid audio.

        :return: True if it's valid, false if not.
        :rtype: bool
        """
        try:
            self.to_numpy()
            return True
        except Exception:
            return False


class Video(Media):
    """Video helpers for models that accept or return video.

    The `Video` class is a wrapper for either binary video content
    or a URL reference to a video file. When an instance contains
    binary video data, it is stored and transferred over HTTP as a
    base64-encoded string. When an instance contains a URL reference,
    the URL is stored and transferred over HTTP as a string. The
    URL representation can be more advantageous when the video file is
    already hosted on a server, as it avoids the overhead of base64
    encoding and decoding.

    The `Video` class contains additional convenience methods to
    read and write video files in common formats.
    """

    @classmethod
    def from_numpy(
        cls,
        video_frames: List[ArrayLike],
        codec="libx264",
        pixel_format="yuv420p",
        fps=30,
    ) -> "Video":
        """Create from NumPy video frames.

        :param video_frames: list of NumPy arrays representing video frames
        :type video_frames: List[ArrayLike]
        :param codec: FFMPEG video codec, defaults to 'libx264'
        :type codec: str, optional
        :param pixel_format: FFMPEG pixel format, defaults to 'yuv420p'
        :type pixel_format: str, optional
        :param fps: frames per second, defaults to 30
        :type fps: int, optional
        :return: Video object
        :rtype: :class:`Video`
        """
        iiov3 = _import_imageio()

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with iiov3.imopen(f.name, "w", plugin="pyav") as vf:
                vf.init_video_stream(codec=codec, fps=fps, pixel_format=pixel_format)

                for frame in video_frames:
                    vf.write_frame(frame)  # type: ignore[arg-type]

            return cast(Video, cls.from_file(f.name))

    def to_numpy_iterator(self) -> Tuple[Iterator[ArrayLike], str]:
        """Convert to a NumPy iterator.

        :return: tuple of iterator of NumPy arrays representing video frames
            and the file name of the temporary file. Users of this method
            should delete the temporary file when the iterator is no longer
            needed to avoid filling up the disk.
        :rtype: Tuple[Iterator[ArrayLike], str]
        """
        iiov3 = _import_imageio()

        # return also file name so user can delete it when they're done
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            self.to_file(f.name)
            return iiov3.imiter(f.name), f.name

    def to_numpy(self) -> ArrayLike:
        """Convert to a NumPy array.

        :return: NumPy array representing video frames
        :rtype: ArrayLike
        """
        iiov3 = _import_imageio()

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            self.to_file(f.name)
            return iiov3.imread(f.name)

    def is_valid(self) -> bool:
        """Check if this is a valid video.

        :return: True if valid, False if invalid
        :rtype: bool
        """
        iiov3 = _import_imageio()
        from imageio.core.request import InitializationError

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            self.to_file(f.name)
            fname = f.name

            try:
                metadata = iiov3.immeta(fname, plugin="pyav")
            except OSError:
                # audio files cause this error
                return False
            except InitializationError:
                # incompatible files cause this error
                return False

            return "video_format" in metadata


# create namespace for future extensions
class File(UploadFile):
    """File class for file uploads in form-data endpoints."""

    pass
