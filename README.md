# Lab 2: Fine-tuning LLM with LoRA

**Course:** ID2223 Scalable Machine Learning 
**Group DFDF** Renda Guo & Zhihan Zhao


## Overview

Fine-tuned Llama-3.2-3B-Instruct on the [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset using QLoRA (4-bit quantization + LoRA) with Unsloth.

- **Base Model:** unsloth/Llama-3.2-3B-Instruct
- **Dataset:** mlabonne/FineTome-100k
- **Method:** QLoRA with 4-bit quantization
- **UI:** [Gradio Chatbot on HuggingFace Spaces](https://huggingface.co/spaces/Ghostzh/your-space-name)

## Improvement Strategies

### Potential Improvement

| Parameter | Current | Potential Improvement |
|-----------|---------|----------------------|
| LoRA rank (r) | 16 | Try 32 or 64 for more capacity |
| LoRA alpha | 16 | Increase to 32 with higher rank |
| Learning rate | 2e-4 | Experiment with 1e-4 or 5e-5 |
| Epochs | 1 | Train for 2-3 epochs |
| Batch size | 2 | Increase if GPU memory allows |

Other improvements:
- Use Llama-3.2-1B for faster inference on CPU
- Try different base models (Mistral, Phi-3, Gemma-2)
- Enable packing for faster training on short sequences

## How to Run

### Training (Google Colab)
1. Open the notebook in Colab with T4 GPU
2. Run all cells to fine-tune the model
3. Checkpoints are saved to Google Drive automatically

### Inference (HuggingFace Spaces)
Visit: [Chatbot UI Link](https://huggingface.co/spaces/Ghostzh/your-space-name)

## Files

- `Llama_3_2_finetuning.ipynb` - Training notebook
- `app.py` - Gradio chatbot interface
- `requirements.txt` - Dependencies for HuggingFace Space
