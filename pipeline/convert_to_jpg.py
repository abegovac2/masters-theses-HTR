from PIL import Image
import os


class ConvertImageToJpeg:

    def __init__(self) -> None:
        pass

    def execute(self, src: str) -> str:
        if src.endswith(".jpg"):
            return src

        if not src.endswith((".png", ".gif")):
            return None

        suffix = src.split(".")[-1]
        dest = f"{src.removesuffix(suffix)}.jpg"
        with Image.open(src) as img:
            img.convert("RGB").save(dest, format="JPEG", quality=95)

        os.remove(src)
        return dest
