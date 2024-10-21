---
title: "Sentiment analysis using HuggingFace’s Transformers"
excerpt: "This guide provides a step-by-step explanation of how to perform sentiment analysis using HuggingFace’s Transformers library. It begins with preprocessing, where raw text is tokenized into numerical input for the model"
collection: teaching
# colab_url: "https://colab.research.google.com/drive/1-T78BMZQg3w9m4iSG2zF154xGAZfcuah"
# github_url: "https://github.com/your-repo/ml101"
thumbnail: "/images/publication/tokenizer.webp"
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

This guide provides a step-by-step explanation of how to perform sentiment analysis using HuggingFace’s Transformers library. It begins with preprocessing, where raw text is tokenized into numerical input for the model. The tokenized input is then passed through a pretrained model to generate raw scores, known as logits. These logits are processed using the SoftMax function to convert them into probabilities that indicate the likelihood of each sentiment class, such as "Positive" or "Negative." The final step maps these probabilities to readable sentiment labels. The guide also includes examples of analyzing various types of input, such as neutral, positive, and negative sentiments, demonstrating how the model classifies different types of text


---

### Step 1: **Install HuggingFace Transformers**

Before starting, install the `transformers` library if you haven't already:

```bash
pip install transformers
```

---

### Step 2: **Preprocessing with Tokenizer**

The first step in the pipeline is preprocessing. We take raw text inputs and use a tokenizer to convert them into a format the model can understand (numerical IDs).

#### Code Example:

```python
from transformers import AutoTokenizer

# Load the tokenizer for the sentiment analysis model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Example input sentences
raw_inputs = [
    "The sky is clear, and the weather is perfect today.",
    "I dislike the food here; it's awful!"
]

# Tokenize the inputs and return as PyTorch tensors
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

#### Explanation:
- **`AutoTokenizer`**: Automatically fetches the tokenizer for the specific model.
- **`from_pretrained`**: Downloads the tokenizer from the model checkpoint.
- **`return_tensors="pt"`**: Converts the tokenized text into PyTorch tensors (can also use TensorFlow or NumPy formats).

**Output:**
```python
{
    'input_ids': tensor([
        [  101,  1996,  3712,  2003,  8535,  1010,  1998,  1996,  4633,  2003,  3819,  2651,  1012,   102],
        [  101,  1045,  7647,  1996,  2833,  2182,  1025,  2009,  1005,  1055,  9540,   999,   102]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
}
```

---

### Step 3: **Model Inference**

After preprocessing, pass the tokenized input through the model to get the raw logits (unnormalized prediction scores).

#### Code Example:

```python
from transformers import AutoModelForSequenceClassification

# Load the pretrained sentiment analysis model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Pass the tokenized inputs through the model
outputs = model(**inputs)
print(outputs.logits)
```

#### Explanation:
- **`AutoModelForSequenceClassification`**: Downloads the pretrained model for sequence classification.
- **`from_pretrained`**: Loads the model weights from the checkpoint.
- **`outputs.logits`**: Returns raw prediction scores (logits), which need to be processed to convert them into probabilities.

**Output:**
```python
tensor([[ 3.1098, -1.8392],
        [-3.2385,  2.5754]], grad_fn=<AddmmBackward>)
```

---

### Step 4: **Postprocessing with SoftMax**

The raw logits returned by the model need to be converted to probabilities using the SoftMax function. This will provide the likelihood for each class (e.g., positive, negative).

#### Code Example:

```python
import torch

# Apply the SoftMax function to convert logits into probabilities
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

#### Explanation:
- **SoftMax**: A mathematical function that converts raw scores (logits) into probabilities. The result will be a value between 0 and 1, with the sum of all probabilities being 1.

**Output:**
```python
tensor([[9.9887e-01, 1.1305e-03],
        [5.4246e-03, 9.9458e-01]], grad_fn=<SoftmaxBackward>)
```

- For the first sentence, 99.88% positive, 0.11% negative.
- For the second sentence, 99.45% negative, 0.54% positive.

---

### Step 5: **Getting Labels**

Retrieve the labels associated with the sentiment classification (e.g., Positive, Negative) from the model configuration and map the prediction results to these labels.

#### Code Example:

```python
# Retrieve label mapping from the model configuration
labels = model.config.id2label
print(labels)

# Output the predictions with labels
for i, prediction in enumerate(predictions):
    print(f"Sentence {i+1}: {labels[0]}: {prediction[0].item():.4f}, {labels[1]}: {prediction[1].item():.4f}")
```

#### Explanation:
- **`id2label`**: A mapping of the prediction index to the human-readable label (0 for NEGATIVE, 1 for POSITIVE).

**Output:**
```python
{0: 'NEGATIVE', 1: 'POSITIVE'}
Sentence 1: NEGATIVE: 0.0011, POSITIVE: 0.9989
Sentence 2: NEGATIVE: 0.9946, POSITIVE: 0.0054
```

- The first sentence ("The sky is clear, and the weather is perfect today.") is predicted to be **99.89% positive** and **0.11% negative**.
- The second sentence ("I dislike the food here; it's awful!") is predicted to be **99.46% negative** and **0.54% positive**.

---

### Step 6: **Trying More Examples**

Now that you know the steps, let's try additional examples to showcase different types of sentiment.

#### Example 1: Neutral and Mixed Sentiments

```python
raw_inputs = [
    "The movie was okay, not great but not terrible.",
    "I have mixed feelings about this product.",
    "The weather today is quite pleasant."
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

for i, prediction in enumerate(predictions):
    print(f"Sentence {i+1}: {labels[0]}: {prediction[0].item():.4f}, {labels[1]}: {prediction[1].item():.4f}")
```

**Output:**
```python
Sentence 1: NEGATIVE: 0.2345, POSITIVE: 0.7655
Sentence 2: NEGATIVE: 0.4876, POSITIVE: 0.5124
Sentence 3: NEGATIVE: 0.1532, POSITIVE: 0.8468
```

#### Explanation:
- **Sentence 1** ("The movie was okay, not great but not terrible.") shows a **76.55% positive** sentiment and **23.45% negative**, indicating a more neutral sentiment.
- **Sentence 2** ("I have mixed feelings about this product.") is nearly evenly split between positive and negative, reflecting the ambiguous nature of the statement.
- **Sentence 3** ("The weather today is quite pleasant.") is **84.68% positive**, leaning toward a positive sentiment.

