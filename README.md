# bhagwadgita_patanjiliyoga

processed_data/  # Optional, for preprocessed data
 models/
 bert_finetuned/  # Fine-tuned BERT model weights
 gpt_neo_finetuned/  # Fine-tuned GPT-Neo model weights
 ## Dataset
- Bhagavad Gita dataset with questions, Sanskrit verses, and English translations.
- Yoga Sutras dataset containing structured information and QA pairs.
  ### Prerequisites
- Python 3.8+
- GPU (optional but recommended for faster model inference).
  ## requirements.txt
  torch>=1.12.0
transformers>=4.21.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0  # Replace with faiss-gpu if using GPU
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.0.0
regex>=2022.1.0
jupyter>=1.0.0
tqdm>=4.62.0
pytest>=7.0.0
matplotlib>=3.5.0
seaborn>=0.11.2
 pip install -r requirements.txt
