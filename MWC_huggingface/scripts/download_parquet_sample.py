#!/usr/bin/env python3
"""
Download parquet files directly from MLCommons/peoples_speech dataset
For testing the training pipeline
"""

import os
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_sample_parquet():
    """Download parquet file directly (no audio decoding needed)"""
    
    # Target directory
    output_dir = Path(__file__).resolve().parent.parent / "data" / "audio" / "parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # HuggingFace dataset info - microset is smallest for testing
    repo_id = "MLCommons/peoples_speech"
    filename = "microset/train-00000-of-00001.parquet"
    
    logger.info("="*80)
    logger.info("DOWNLOADING PARQUET FILE FROM PEOPLES_SPEECH")
    logger.info("="*80)
    logger.info(f"Repository: {repo_id}")
    logger.info(f"File: {filename}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Download the parquet file directly
        logger.info("\nDownloading parquet file...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"\n✅ Downloaded successfully!")
        logger.info(f"File location: {downloaded_path}")
        
        # Get file size
        file_size = os.path.getsize(downloaded_path) / (1024 * 1024)  # MB
        logger.info(f"File size: {file_size:.2f} MB")
        
        # Quick inspection
        try:
            import pandas as pd
            df = pd.read_parquet(downloaded_path)
            logger.info(f"\nDataset info:")
            logger.info(f"  Rows: {len(df)}")
            logger.info(f"  Columns: {list(df.columns)}")
            
            if 'audio' in df.columns:
                logger.info(f"  ✓ Audio column found")
                logger.info(f"    Audio data type: {type(df['audio'].iloc[0])}")
            if 'text' in df.columns:
                logger.info(f"  ✓ Text column found")
                logger.info(f"\nFirst transcription:")
                logger.info(f"  {df['text'].iloc[0][:150]}...")
        except Exception as e:
            logger.warning(f"Could not inspect parquet: {e}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ READY FOR TRAINING!")
        logger.info("="*80)
        logger.info(f"Parquet directory: {output_dir}")
        logger.info(f"\nYour config is already set to:")
        logger.info(f"  file_path: \"{output_dir}/\"")
        
        return downloaded_path
        
    except Exception as e:
        logger.error(f"\n❌ Download failed: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check your internet connection")
        logger.error("  2. Install: pip install huggingface_hub")
        logger.error("  3. Try again in a few minutes")
        raise


if __name__ == "__main__":
    download_sample_parquet()
