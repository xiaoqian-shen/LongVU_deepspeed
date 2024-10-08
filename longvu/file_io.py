# Copyright (c) Meta Platforms, Inc. and its affiliates.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from iopath.common.file_io import HTTPURLHandler, PathManager as PathManagerBase

__all__ = ["PathManager"]


PathManager = PathManagerBase()
PathManager.register_handler(HTTPURLHandler())
