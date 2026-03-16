"""
init_s3.py — One-time script to upload reference data to S3.
Run once before starting the API for the first time.
"""

import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

DISEASES = {
    "colomerus_vitis":             "Erinose",
    "elsinoe_ampelina":            "Anthracnose",
    "erysiphe_necator":            "Powdery mildew",
    "guignardia_bidwellii":        "Black rot",
    "healthy":                     "Healthy",
    "phaeomoniella_chlamydospora": "Esca",
    "plasmopara_viticola":         "Downy mildew"
}

def upload_diseases(bucket_name: str, dataset_name: str) -> None:
    """Uploads the disease label dictionary to S3."""
    s3 = boto3.client('s3')
    key = f'vitiscan-data/disease-{dataset_name}.json'
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(DISEASES, ensure_ascii=False).encode('utf-8')
    )
    print(f"Uploaded to s3://{bucket_name}/{key}")

if __name__ == "__main__":
    bucket = os.getenv("S3_BUCKET_NAME", "vitiscanpro-bucket")
    upload_diseases(bucket, "inrae")