# Encoding chunks on Kaggle

The laptop CPU takes about 45 minutes per chunking config to encode the mini corpus, and SPLADE needs more RAM than it has free. A Kaggle T4 does every config in about 10–15 minutes. Queries are always encoded on the CPU with fastembed, and `parity.json` checks that both backends agree.

1. Build the bundle (about 2.5 MB): `python kaggle/bundle.py` writes `kaggle/bundle.zip`.
2. On kaggle.com, create a **New Dataset**, upload `bundle.zip`, and give it the slug `blended-rag-bundle`.
3. Create a **New Notebook**, add that dataset, and set Accelerator to **GPU T4 x2** and Internet to **On**.
4. Run one cell:
   ```
   !pip install -q fastembed
   !python /kaggle/input/blended-rag-bundle/kaggle/encode.py --out /kaggle/working
   ```
5. Download `/kaggle/working/vectors.zip` from the Output panel and unzip it into `artifacts/vectors/`.
