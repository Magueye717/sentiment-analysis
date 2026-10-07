# SentimentScope: Sentiment Analysis with a Transformer

This project trains a small GPT-style transformer from scratch in PyTorch to classify IMDB movie reviews as positive or negative.

## What it does
- Loads and explores the IMDB dataset (25,000 training and 25,000 test reviews)
- Tokenizes reviews with the `bert-base-uncased` tokenizer (max length 128)
- Builds a custom `Dataset` and `DataLoader`s
- Adapts the `DemoGPT` model for classification with mean pooling and a linear head
- Trains with cross-entropy loss and the AdamW optimizer, then evaluates on the test set

## Results
- Validation accuracy before training: 49.72% (random weights)
- Validation accuracy after 3 epochs: 78.64%
- Test accuracy: 77.33% (project target: above 75%)

## Dependencies
- Python 3
- torch
- transformers
- pandas
- numpy
- matplotlib

Install them with:

    pip install torch transformers pandas numpy matplotlib

## How to run
1. Download `aclImdb_v1.tar.gz` and place it in the same folder as the notebook.
2. Open `SentimentScope.ipynb` and run all cells from top to bottom. The first cell extracts the dataset.
3. Use a GPU for training if one is available. It is much faster.

## Files
- `SentimentScope.ipynb`: the notebook with the full code, outputs and conclusion 
