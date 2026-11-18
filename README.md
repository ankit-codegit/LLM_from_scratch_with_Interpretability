\# LLM from Scratch with Interpretability



A decoder-only GPT-style language model implemented using PyTorch, covering the complete workflow from tokenization and data preparation to model architecture, training, inference, and interpretability.



The goal of this project is to understand how GPT-style language models work internally by implementing their core components and exploring how information flows through transformer layers and attention heads.



\## Project Overview



This project implements a GPT-style Transformer language model from scratch and explores its internal behavior through interpretability techniques.



The implementation covers:



\- Text tokenization and embedding

\- Input sequence preparation and data loading

\- Scaled dot-product self-attention

\- Multi-head causal self-attention

\- Transformer blocks and GPT architecture

\- Language model training and checkpointing

\- Autoregressive text generation

\- KV caching during inference

\- Token-, layer-, and attention-head-level interpretability



\## Architecture



The model follows a decoder-only Transformer architecture.



```text

&#x20;               Input Text

&#x20;                   |

&#x20;                   v

&#x20;               Tokenizer

&#x20;                   |

&#x20;                   v

&#x20;             Token Embeddings

&#x20;                   +

&#x20;            Positional Embeddings

&#x20;                   |

&#x20;                   v

&#x20;            Transformer Blocks

&#x20;                   |

&#x20;         +---------+---------+

&#x20;         |                   |

&#x20;         v                   |

&#x20;    Layer Normalization      |

&#x20;         |                   |

&#x20;         v                   |

&#x20;   Causal Self-Attention     |

&#x20;         |                   |

&#x20;         v                   |

&#x20;     Residual Connection <---+

&#x20;         |

&#x20;         v

&#x20;    Layer Normalization

&#x20;         |

&#x20;         v

&#x20;     Feed Forward

&#x20;         |

&#x20;         v

&#x20;     Residual Connection

&#x20;         |

&#x20;         v

&#x20;      Final LayerNorm

&#x20;         |

&#x20;         v

&#x20;       LM Head

&#x20;         |

&#x20;         v

&#x20;   Next Token Prediction

```



\## Project Structure



The repository contains six Jupyter notebooks organized around different components of the language model.



| Notebook | Description |

|---|---|

| `Tokenizer+embedding+Data\_loader.ipynb` | Tokenization, token embeddings, positional embeddings, and data loading |

| `Attention\_block.ipynb` | Understanding and implementing the attention mechanism |

| `GPT\_architecture.ipynb` | Transformer blocks, normalization, residual connections, and GPT architecture |

| `GPT\_train.ipynb` | Dataset preparation, training pipeline, optimization, and checkpoint management |

| `GPT\_inference\_model.ipynb` | Loading the model, autoregressive generation, and inference |

| `GPT\_interpretability.ipynb` | Exploring model predictions, internal representations, and attention-head behavior |



\## Dataset and Tokenization



The training pipeline explores a combination of text sources:



\- CSCorpus: A curated computer science corpus.

\- FineWeb-Edu: Educational web text.

\- WikiText-103: A language modeling dataset.



The implementation uses the GPT-2 tokenizer through `tiktoken`.



The training pipeline includes:

\- Text preprocessing and tokenization

\- Train/validation data preparation

\- Binary token storage

\- Memory-mapped dataset loading

\- Batch generation for model training



\## Training



The model is trained using autoregressive next-token prediction.



Given a sequence of previous tokens, the model learns to predict the next token.



The training implementation includes:



\- Cross-entropy language modeling loss

\- AdamW optimizer

\- Learning-rate warmup and cosine decay

\- Training and validation loss estimation

\- Model and optimizer checkpointing



\## Inference



The inference implementation supports autoregressive text generation.



It includes:

\- Loading trained model checkpoints

\- Tokenizing input prompts

\- Generating tokens sequentially

\- Causal attention during generation

\- Key-Value (KV) caching to reuse previously computed attention keys and values



\## Interpretability



An important focus of this project is understanding not just what the model predicts, but how its internal computations contribute to those predictions.



The interpretability notebook explores three levels of analysis.



\### 1. Token-Level Interpretability



Investigating how input tokens influence the model's predictions.



Techniques explored:

\- Final prediction distribution

\- Leave-one-out token contribution

\- Layer evolution using the Logit Lens

\- Confidence evolution across layers



\### 2. Layer-Level Interpretability



Investigating how information evolves across transformer layers.



Areas explored include:

\- Part-of-speech (POS) information

\- Named Entity Recognition (NER)

\- Sentiment

\- Topic information

\- Intermediate hidden representations



\### 3. Attention-Head-Level Interpretability



Exploring different attention-head behaviors, including:



\- Previous-token attention

\- Induction heads

\- Name-mover behavior

\- Long-range dependency tracking

\- Structural pattern tracking



The project also captures attention weights and intermediate query, key, and value representations for analysis.



\## Technologies Used



\- \*\*Python\*\* – Core programming language

\- \*\*PyTorch\*\* – Neural network implementation and training

\- \*\*NumPy\*\* – Numerical operations and data processing

\- \*\*tiktoken\*\* – GPT-2 tokenization

\- \*\*Hugging Face Datasets\*\* – Dataset loading and streaming

\- \*\*Jupyter Notebook / Google Colab\*\* – Development and experimentation



\## Getting Started



\### 1. Clone the repository



```bash

git clone https://github.com/ankit-codegit/LLM\_from\_scratch\_with\_Interpretability.git

cd LLM\_from\_scratch\_with\_Interpretability

```



\### 2. Install dependencies



```bash

pip install torch numpy tiktoken datasets tqdm pyarrow

```



Install any additional packages required by individual notebooks.



\### 3. Open the notebooks



The notebooks can be opened using Jupyter Notebook or Google Colab.



For Google Colab, upload the notebooks or open them directly from the GitHub repository.



\### 4. Prepare the training data



The training notebook contains the data preparation pipeline. Configure the dataset paths and Google Drive directories according to your environment.



\### 5. Train and run inference



Execute the training notebook to prepare the dataset and train the model.



Save the resulting checkpoints and configure the inference notebook to load the corresponding checkpoint.



\*\*Note:\*\* Dataset files and trained model checkpoints are not included in this repository. Training and inference require the corresponding data and checkpoint files to be prepared separately.



\## Learning Objectives



This project explores the implementation and internal workings of GPT-style language models, with particular emphasis on:



\- Understanding Transformer architecture from first principles

\- Implementing attention mechanisms using PyTorch

\- Building an autoregressive language modeling pipeline

\- Understanding model training and inference

\- Investigating how transformer layers and attention heads process information



\---



\*\*Author:\*\* Ankit



\*\*Repository:\*\* \[LLM\_from\_scratch\_with\_Interpretability](https://github.com/ankit-codegit/LLM\_from\_scratch\_with\_Interpretability)



