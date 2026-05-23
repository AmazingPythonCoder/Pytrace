from __future__ import annotations

import numpy as np
from OpenGL import GL


class ImageTexture:
    """Small OpenGL texture wrapper for ImGui image previews."""

    def __init__(self) -> None:
        self.texture = 0
        self.size = (0, 0)

    def update_rgb(self, rgb: np.ndarray) -> int:
        data = np.asarray(rgb, dtype=np.uint8)
        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError("ImageTexture expects HxWx3 uint8 data")
        height, width = int(data.shape[0]), int(data.shape[1])
        if not self.texture:
            self.texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB8,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            np.ascontiguousarray(data),
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        self.size = (width, height)
        return self.texture

    def cleanup(self) -> None:
        if self.texture:
            GL.glDeleteTextures([self.texture])
        self.texture = 0
        self.size = (0, 0)
