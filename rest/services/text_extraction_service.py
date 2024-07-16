from src.data.generator import Tokenizer
from src.network.model import HTRModel
from src.data.preproc import preprocess, normalization
from rest.models.model import Detection
import cv2


class TextExtractionService:
    charset_base = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ ČčĆćĐđŽžŠš"

    input_size = (1024, 128, 1)
    max_text_length = 256
    arch = "flor"
    model = None

    def __init__(self, model_path) -> None:
        self.tokenizer = Tokenizer(
            chars=self.charset_base, max_text_length=self.max_text_length
        )

        self.model = HTRModel(
            architecture=self.arch,
            input_size=self.input_size,
            vocab_size=self.tokenizer.vocab_size,
            beam_width=30,
            top_paths=10,
        )

        self.model.compile(learning_rate=0.001)
        self.model.load_checkpoint(target=model_path)

    def extract(self, detection: Detection):
        image = detection.line_image.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = preprocess(image, input_size=self.input_size)
        x_test = normalization([image])
        predictions, probabilities = self.model.predict(x_test, ctc_decode=True)
        predictions = [[self.tokenizer.decode(x) for x in y] for y in predictions]
        predict = [(el[0][0], el[1][0]) for el in list(zip(predictions, probabilities))]
        text, probability = predict[0]
        detection.text = text
        detection.certanty = round(float(probability), 4)
        return detection
