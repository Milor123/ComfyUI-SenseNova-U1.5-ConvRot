from .nodes import comfy_entrypoint
from .qt_guards import install_quant_guards

install_quant_guards()

WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
