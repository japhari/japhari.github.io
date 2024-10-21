---
title: "How to create and use Transformer models with HuggingFace's AutoModel"
excerpt: "This section explains how to create and use Transformer models with HuggingFace's AutoModel and other related classes. The AutoModel class is a convenient wrapper that can automatically determine and load the appropriate model architecture based on a checkpoint"
collection: teaching
# colab_url: "https://colab.research.google.com/drive/1-T78BMZQg3w9m4iSG2zF154xGAZfcuah"
# github_url: "https://github.com/your-repo/ml101"
thumbnail: "/images/publication/huggingface.png"
type: "Natural Language Processing"
permalink: /teaching/2014-spring-teaching-1
venue: "PTIT , Department of Computer Science"
date: 2024-10-21
location: "Hanoi, Vietnam"
categories:
  - teaching
tags:
  - nlp
  - transformers
---

This section explains how to create and use Transformer models with HuggingFace's `AutoModel` and other related classes. The `AutoModel` class is a convenient wrapper that can automatically determine and load the appropriate model architecture based on a checkpoint. However, if you already know which model type you need, such as BERT, you can directly instantiate it using its respective class like `BertModel`.

1. **Creating a Transformer Model**: You can initialize a model with a configuration using `BertConfig`. This configuration contains various model attributes such as hidden size, number of layers, and attention heads. Initially, the model is randomly initialized and needs to be trained.

2. **Loading Pretrained Models**: To avoid training from scratch, which is time-consuming and resource-intensive, you can load a pretrained model using the `from_pretrained()` method. This allows you to reuse models trained by others, like the popular BERT model (`bert-base-cased`). Using `AutoModel` instead of a specific model class (like `BertModel`) makes your code more checkpoint-agnostic, meaning it can adapt to different architectures.

3. **Saving Models**: After using or fine-tuning a model, you can save it with the `save_pretrained()` method, which saves both the configuration (`config.json`) and model weights (`pytorch_model.bin`). These files can be used to reload the model later.

4. **Using Models for Inference**: Models require input in the form of tokenized numbers (input IDs). Tokenizers convert sequences like "Hello!" into a list of integers that can be fed into the model as tensors. Once the input is tokenized and transformed into tensors, you can pass it to the model for predictions.


---

Here’s a summary with code examples demonstrating different use cases for creating, loading, saving, and using Transformer models with HuggingFace.

---

### Example 1: **Creating a Transformer Model from Configuration**

In this example, we create a BERT model using the `BertConfig` and `BertModel` classes.

```python
from transformers import BertConfig, BertModel

# Create a BERT configuration
config = BertConfig()

# Initialize a BERT model using the configuration
model = BertModel(config)

# Print the configuration details
print(config)
```

**Output:**

```python
BertConfig {
  "hidden_size": 768,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  ...
}
```

---

### Example 2: **Loading a Pretrained Model**

Instead of training from scratch, we load a pretrained BERT model using the `from_pretrained()` method.

```python
from transformers import BertModel

# Load a pretrained BERT model
model = BertModel.from_pretrained("bert-base-cased")

# Print the model architecture
print(model)
```

**Output:**
```python
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(28996, 768, padding_idx=0)
    ...
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0): BertLayer(...)
      ...
    )
  )
  (pooler): BertPooler(...)
)
```

---

### Example 3: **Using `AutoModel` for Flexible Loading**

To make the code more flexible and adaptable to different checkpoints and architectures, use `AutoModel`.

```python
from transformers import AutoModel

# Load a model using AutoModel (architecture-agnostic)
model = AutoModel.from_pretrained("bert-base-cased")

# Print the model details
print(model)
```

This ensures that the model is automatically loaded based on the checkpoint you specify.

---

### Example 4: **Saving a Model**

Once a model is loaded or fine-tuned, you can save it to a specific directory using the `save_pretrained()` method.

```python
# Save the model to a directory
model.save_pretrained("my_model_directory")

# Check the contents of the directory
!ls my_model_directory
```

**Output:**
```bash
config.json    pytorch_model.bin
```

---

### Example 5: **Using a Model for Inference**

Here’s an example of how to use a BERT model for making predictions by converting text into tokens (numbers) that the model can process.

```python
from transformers import BertTokenizer
import torch

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

# Example input text
sequences = ["Hello!", "Cool.", "Nice!"]

# Convert text into token IDs
encoded_sequences = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)

# Pass the encoded inputs to the model
output = model(**encoded_sequences)

# Print the model output
print(output)
```

**Output:**

```python
BaseModelOutputWithPoolingAndCrossAttentions(
    last_hidden_state=tensor([[[-0.0505,  0.0244,  0.0584,  ..., -0.1566,  0.0499,  0.0146],
         [-0.0901, -0.0601,  0.0582,  ..., -0.2255, -0.0472,  0.1543],
         [-0.0577,  0.0153,  0.1073,  ..., -0.1121,  0.0168,  0.1146]],
        ...
    pooler_output=tensor([[-0.6588, -0.5088,  0.6350,  ..., -0.0093,  0.3687,  0.4415]]), 
)
```

---

### Example 6: **Converting Text to Tokens**

The tokenizer is responsible for converting text into token IDs. Here's an example of how to tokenize text for inference.

```python
# Example text input
sequences = ["Hello!", "Cool.", "Nice!"]

# Tokenize the sequences into input IDs
encoded_sequences = tokenizer(sequences)

# Print the tokenized input (IDs)
print(encoded_sequences["input_ids"])
```

**Output:**

```python
[[101, 7592, 999, 102], [101, 4658, 1012, 102], [101, 3835, 999, 102]]
```

These are the token IDs for the input text, which the model will use for inference.

---

### Summary

- **Creating Models**: You can instantiate a model using its configuration or load a pretrained one.
- **Loading Models**: Pretrained models can be loaded with the `from_pretrained()` method.
- **Saving Models**: Save models to a directory using `save_pretrained()`.
- **Inference**: Convert text to tokens using a tokenizer and pass them to the model for predictions.
- **Using AutoModel**: Make your code flexible and adaptable to various architectures by using `AutoModel`.

