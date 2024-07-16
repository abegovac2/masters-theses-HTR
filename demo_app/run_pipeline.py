import os
import cv2
from uuid import uuid4

from pipeline.convert_to_jpg import ConvertImageToJpeg
from utils.clean_image_pipeline import apply_image_cleaning_pipeline
from pipeline.extract_regions_yolo import ExtractDocumentRegions
from pipeline.extract_words_yolo import ExtractWordsYolo
from pipeline.handwritten_text_recognition import ExtractText

from concurrent.futures import ThreadPoolExecutor


class TextDetectionPipeline:
    HELPER_DIR = "helper_dir"

    def __init__(self) -> None:

        self._convert_to_jpeg = ConvertImageToJpeg()
        self._ext_doc_regions = ExtractDocumentRegions()
        self._ext_word_yolo = ExtractWordsYolo()
        self._ext_text = ExtractText()

    def _execute(self, src, interesting_region=5):
        if isinstance(interesting_region, int):
            interesting_region = [interesting_region]
        uuid = str(uuid4())

        name = src.split("/")[-1]
        name = name.removesuffix(".jpg")

        img_path = self._convert_to_jpeg.execute(src)
        image = apply_image_cleaning_pipeline(img_path, img_path, save=False)

        image_clean = f"./{self.HELPER_DIR}/{name}_clean_{uuid}.jpg"

        cv2.imwrite(image_clean, image)
        image = cv2.imread(image_clean)

        imgs = self._ext_doc_regions.execute(image, interesting_region)

        ############
        def thread_work(image, region):

            words = self._ext_word_yolo.execute(image)

            images = []
            for i, word in enumerate(words):
                img_name = f"./{self.HELPER_DIR}/{name}_{i}_{region}_{uuid}.jpg"
                images.append(img_name)
                cv2.imwrite(img_name, word)

            return list(
                zip(images, [self._ext_text.execute(img)[0][0] for img in images])
            )

        ############
        results = []
        threads = []
        with ThreadPoolExecutor() as tpe:
            for region, image in imgs:
                threads.append(tpe.submit(thread_work, image, region))

            results = [thread.result() for thread in threads]

        return results

    def execute(self, src, interesting_region=5):
        try:
            if not os.path.exists(self.HELPER_DIR):
                os.makedirs(self.HELPER_DIR)

            return self._execute(src, interesting_region)
        finally:
            images = os.listdir(self.HELPER_DIR)
            for img in images:
                path = os.path.join(self.HELPER_DIR, img)
                os.remove(path)


def cleanup_images(tdp: TextDetectionPipeline):

    yield


"""
a = TextDetectionPipeline().execute(
    "./jpg_minski_zapisnici/FormatA/30535.jpg", [1, 2, 3, 4, 5, 6, 7, 8]
)
print(a)
"""
