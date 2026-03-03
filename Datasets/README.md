
## 📥 Datasets

### Option 1: Download from HuggingFace (Recommended)

Download datasets manually from HuggingFace and place them in `Datasets/raw/`:

```bash
# Create raw data directory
mkdir -p Datasets/raw

# Download HotpotQA
cd Datasets/raw
git clone https://huggingface.co/datasets/hotpotqa/hotpot_qa hotpotqa

# Download UltraDomain
git clone https://huggingface.co/datasets/TommyChien/UltraDomain ultradomain

# Download 2WikiMultihopQA
git clone https://huggingface.co/datasets/framolfese/2WikiMultihopQA 2wikimultihop

# Download MuSiQue
git clone https://huggingface.co/datasets/bdsaglam/musique musique

cd ../..
```

### Option 2: Download via Script

```bash
# Download from network (may encounter path length issues on Windows)
python Datasets/download_datasets.py --datasets hotpotqa --download

# List supported datasets
python Datasets/download_datasets.py --list
```

### Convert Datasets

After downloading, convert raw data to benchmark format:

```bash
# Convert from local raw data (default)
python Datasets/download_datasets.py --datasets hotpotqa
python Datasets/download_datasets.py --datasets all

# Output: Datasets/Corpus/ and Datasets/Questions/
```

### Dataset Structure

```
Datasets/
├── raw/                        # Raw downloaded data
│   ├── hotpotqa/
│   ├── ultradomain/
│   └── ...
├── Corpus/                     # Parsed corpus (for frameworks)
│   ├── hotpotqa.parquet
│   └── ...
└── Questions/                  # Parsed questions (for frameworks)
    ├── hotpotqa_questions.parquet
    └── ...
```