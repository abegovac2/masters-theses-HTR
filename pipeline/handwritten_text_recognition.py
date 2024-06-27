from src.data.generator import Tokenizer
from src.network.model import HTRModel
from src.data.preproc import preprocess, normalization


class ExtractText:

    charset_base = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ ČčĆćĐđŽžŠš"

    input_size = (1024, 128, 1)
    max_text_length = 256
    arch = "flor"

    def __init__(self, model_path="./models/text_detection_model.hdf5") -> None:
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

    def execute(self, img):
        img = preprocess(img, input_size=self.input_size)
        x_test = normalization([img])
        predicts, probabilities = self.model.predict(x_test, ctc_decode=True)
        predicts = [[self.tokenizer.decode(x) for x in y] for y in predicts]

        return [(el[0][0], el[1][0]) for el in list(zip(predicts, probabilities))]
