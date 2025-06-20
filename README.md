# Named Entity Recognition (NER) using HuggingFace Transformers

## Overview

This project demonstrates how to perform Named Entity Recognition (NER) using the HuggingFace `transformers` library and the `datasets` library. The NER task identifies named entities in text (like persons, organizations, locations, etc.) and classifies them into predefined categories. The dataset used is the CoNLL-2003 dataset from HuggingFace's datasets hub.

![NER Task Illustration](https://huggingface.co/datasets/conll2003/resolve/main/docs/images/conll2003-overview.png)

## Dataset

* **Name:** CoNLL-2003 (via `tomaarsen/conllpp`)
* **Fields:** `tokens` (words) and `ner_tags` (integer-encoded labels)
* The dataset is divided into train, test, and validation sets.

## Project Structure & Workflow

### 1. Setup

* Installs the necessary libraries: `transformers`, `accelerate`, `datasets`
* Disables Weights & Biases tracking

```python
import os
os.environ["WANDB_DISABLED"] = "true"
```

### 2. Data Loading & Preprocessing

* Loads the dataset with `datasets.load_dataset`
* Retrieves tokens and associated NER tags
* Maps tag indices to tag names for easier interpretation
* Adds a human-readable `ner_tag_str` field using a custom `create_tag_names` function

```python
from datasets import load_dataset
raw_data = load_dataset("tomaarsen/conllpp")
```

### 3. Tokenization & Label Alignment

* Uses `AutoTokenizer` (specifically `bert-base-cased`) for tokenization
* Aligns the original NER tags with tokenized input, considering sub-word tokenization (using `word_ids()`)

![Tokenization Diagram](https://huggingface.co/blog/assets/02_how-to-train/tokenization.png)

### 4. Model Training

* Model: `AutoModelForTokenClassification` with the number of labels matching the dataset
* Trainer setup includes:

  * `TrainingArguments` (e.g., batch size, learning rate, epochs)
  * Evaluation strategy set to every epoch
  * Metric: `seqeval` for NER performance
* Dataset is tokenized, labels aligned, and training is performed using `Trainer`

![Training Flow](https://miro.medium.com/v2/resize\:fit:1200/1*L6XnICmwVYi9yeB8vB5xJg.png)

### 5. Evaluation

* The model is evaluated on the validation and test sets
* Accuracy, precision, recall, and F1 scores are computed

```python
from seqeval.metrics import classification_report
print(classification_report(true_labels, pred_labels))
```

### 6. Inference

* Custom function provided to tokenize new text, run inference, and extract predicted NER tags
* Token predictions are mapped back to named entity labels using `id2label`

```python
from transformers import pipeline
ner = pipeline("ner", model=model, tokenizer=tokenizer)
print(ner("HuggingFace is based in New York City."))
```

## Dependencies

```bash
pip install -U transformers accelerate datasets seqeval
```

## Key Files and Functions

* `create_tag_names`: Adds string-based NER tag names to the dataset
* `tokenize_and_align_labels`: Aligns labels with tokenized inputs
* `compute_metrics`: Calculates metrics using `seqeval`
* `ner_pipeline(text)`: Inference function for raw text input

## Results

The trained model achieves competitive results on the CoNLL-2003 validation/test sets. Inference examples show successful identification of named entities.

### Example Output:

![NER Inference Output](https://huggingface.co/blog/assets/07_ner/ner_output.png)

## Future Enhancements

* Incorporate model checkpointing and better logging
* Extend inference to batch processing
* Deploy as an API using FastAPI or Flask
* Visualization of entities in input text

## Acknowledgments

* HuggingFace `transformers` and `datasets`
* `tomaarsen/conllpp` dataset on HuggingFace Hub
* `seqeval` for evaluation metrics

---

This project illustrates an end-to-end pipeline for Named Entity Recognition with state-of-the-art transformer models, showcasing the power of HuggingFace's ecosystem.
