from .render_control import RenderCancelled

__all__ = ["RenderCancelled", "render", "save_png"]


def __getattr__(name: str):
    if name in {"render", "save_png"}:
        from .renderer import render, save_png

        return {"render": render, "save_png": save_png}[name]
    raise AttributeError(name)
