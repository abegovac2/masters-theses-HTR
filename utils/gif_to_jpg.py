from PIL import Image
import os
import threading


class ImageConverter(threading.Thread):
    def __init__(self, input_folder, output_folder, file_name):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.file_name = file_name

    def run(self):
        input_path = os.path.join(self.input_folder, self.file_name)
        output_path = os.path.join(
            self.output_folder, os.path.splitext(self.file_name)[0] + ".jpg"
        )

        # Open the GIF image
        with Image.open(input_path) as img:
            # Convert GIF to JPG
            img.convert("RGB").save(output_path, format="JPEG", quality=95)

        print(f"Converted {self.file_name} to JPG")


def convert_gif_to_jpg(input_folder, output_folder, num_threads=4):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Start threads for conversion
    threads = []
    for file_name in files:
        if file_name.endswith(".gif"):
            thread = ImageConverter(input_folder, output_folder, file_name)
            thread.start()
            threads.append(thread)

            # Limit the number of active threads
            if len(threads) >= num_threads:
                for thread in threads:
                    thread.join()
                threads = []

    # Wait for remaining threads to finish
    for thread in threads:
        thread.join()
