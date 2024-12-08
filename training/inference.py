import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import logging
import pandas as pd
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
import pickle

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Verify CUDA availability
if not torch.cuda.is_available():
    logging.warning("CUDA is not available. Inference will use CPU.")

# Logging script start
logging.info("----- Script Started -----")

# Load test dataset
try:
    test_df = pd.read_csv("test.csv")
    test_df = test_df.dropna(subset=["ocr"])
    data = list(test_df["ocr"].values)
    logging.info(f"Test dataset loaded with {len(data)} samples.")
except FileNotFoundError as e:
    logging.error(f"Test dataset not found: {e}")
    exit(1)
except Exception as e:
    logging.error(f"Error loading test dataset: {e}")
    exit(1)

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt",
        cache_dir="cache",
        max_length=512
    )
    logging.info("Tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading tokenizer: {e}")
    exit(1)

# Load model and pipeline
try:
    ocr_pipeline = pipeline(
        "text2text-generation",
        model="model/checkpoint",
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    logging.info("Model and pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or pipeline: {e}")
    exit(1)

# Perform inference
results = []
batch_size = 8

logging.info("Starting inference...")
try:
    for i in tqdm(range(0, len(data), batch_size), desc="Inference Progress"):
        batch = data[i:i + batch_size]
        batch_results = ocr_pipeline(batch)
        results.extend(batch_results)
    logging.info("Inference completed.")
except Exception as e:
    logging.error(f"Error during inference: {e}")
    exit(1)

# Save raw results
try:
    with open("pred.pkl", "wb") as fp:
        pickle.dump(results, fp)
    logging.info("Raw predictions saved to 'pred.pkl'.")
except Exception as e:
    logging.error(f"Error saving raw predictions: {e}")
    exit(1)

# Process and save predictions
try:
    pred_results = [result["generated_text"] for result in results]

    res = pd.DataFrame(
        zip(test_df["ocr"].values, test_df["correct"].values, pred_results, test_df["font"].values),
        columns=["ocr", "correct", "predicted_text", "font"]
    )
    res.to_csv("prediction.csv", index=False)
    logging.info("Processed predictions saved to 'prediction.csv'.")
except Exception as e:
    logging.error(f"Error processing or saving predictions: {e}")
    exit(1)

logging.info("----- Script Completed Successfully -----")
