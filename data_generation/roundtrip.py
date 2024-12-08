import os
import shutil
import logging
import sys
from csv import writer
from PIL import Image, ImageFont, ImageDraw
import pytesseract as pt
import multiprocessing as mp

# Configurations
TEMP_IMAGE_DIR = "./temp_images"
FONT_DIR = "./fonts"
OUTPUT_MISMATCHED_FILE = "mismatched.csv"
OUTPUT_ALL_FILE = "ocr_error_corrrection_dataset.csv"
IMAGE_WIDTH = 300 # adjust dimension accordingly
IMAGE_HEIGHT = 300
FONT_SIZE = 20
LANGUAGE = "hin" # select language which is supported by tesseract

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ocr_script.log"), logging.StreamHandler()],
)

def get_wrapped_text(text, image_width, font):
    """Wrap text to fit within the image width."""
    lines = []
    current_line = ""
    for word in text.split():
        text_width, _ = font.getsize(current_line + " " + word)
        if text_width <= image_width - 10:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def ocr_image(image_path):
    """Perform OCR on the given image."""
    try:
        ocr_text = pt.image_to_string(image_path, lang=LANGUAGE).replace("\n", " ").strip()
        return [ocr_text, os.path.basename(image_path).split('.')[0]]
    except Exception as e:
        logging.error(f"Error during OCR for {image_path}: {e}")
        return ["", os.path.basename(image_path).split('.')[0]]

def parallelize_ocr(image_paths):
    """Parallelize the OCR process."""
    with mp.Pool(processes=mp.cpu_count()) as pool:
        return pool.map(ocr_image, image_paths)

def generate_image(text, font_path):
    """Generate an image with the given text and font."""
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
        image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        wrapped_text = get_wrapped_text(text, IMAGE_WIDTH, font)
        y_coordinate = 10
        for line in wrapped_text:
            draw.text((10, y_coordinate), line, fill="black", font=font)
            y_coordinate += font.getsize(line)[1]

        output_path = os.path.join(TEMP_IMAGE_DIR, f"{os.path.basename(font_path).split('.')[0]}.png")
        image.save(output_path)
        logging.info(f"Image generated: {output_path}")
    except Exception as e:
        logging.error(f"Error generating image with font {font_path}: {e}")

def parallelize_image_generation(text, font_files):
    """Parallelize the image generation process."""
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(generate_image, [(text, font) for font in font_files])

def write_results_to_csv(results, original_text, mismatched_file, all_file):
    """Write OCR results to the specified CSV files."""
    try:
        with open(mismatched_file, "a") as mismatch, open(all_file, "a") as all_data:
            mismatch_writer = writer(mismatch)
            all_writer = writer(all_data)

            for ocr_text, font_name in results:
                all_writer.writerow([ocr_text, original_text, font_name])
                if ocr_text.strip() != original_text.strip():
                    mismatch_writer.writerow([ocr_text, original_text, font_name])
        logging.info("Results written to CSV files.")
    except Exception as e:
        logging.error(f"Error writing results to CSV: {e}")


def main():
    try:
        # Input text
        if len(sys.argv) < 2:
            logging.error("No text provided. Usage: python script.py '<text>'")
            return

        text = sys.argv[1].strip()
        os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

        # Load font files
        font_files = [
            os.path.join(FONT_DIR, file) for file in os.listdir(FONT_DIR) if file.endswith((".ttf", ".otf"))
        ]

        # Generate images
        logging.info("Generating images...")
        parallelize_image_generation(text, font_files)

        # Perform OCR
        logging.info("Performing OCR...")
        image_paths = [os.path.join(TEMP_IMAGE_DIR, img) for img in os.listdir(TEMP_IMAGE_DIR)]
        ocr_results = parallelize_ocr(image_paths)

        # Write results
        write_results_to_csv(ocr_results, text, OUTPUT_MISMATCHED_FILE, OUTPUT_ALL_FILE)

    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
    finally:
        # Cleanup temporary files
        if os.path.exists(TEMP_IMAGE_DIR):
            shutil.rmtree(TEMP_IMAGE_DIR)
            logging.info("Temporary files cleaned up.")

if __name__ == "__main__":
    main()
