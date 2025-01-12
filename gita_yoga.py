import re
from time import time
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM
import json

# Load BERT model and tokenizer
bert_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForQuestionAnswering.from_pretrained(bert_model_name)

# Load GPT model and tokenizer
gpt_model_name = "EleutherAI/gpt-neo-125M"
gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# Function to load and process datasets
def load_and_combine_datasets(file_paths):
    combined_questions = []
    combined_responses = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["question"])
        questions = df["question"].str.lower().astype(str).tolist()

        if "speaker" in df.columns:
            df["formatted_response"] = (
                "Speaker: " + df["speaker"] + "\n"
                "Sanskrit: " + df.get("sanskrit", "N/A") + "\n"
                "Translation: " + df.get("translation", "N/A") + "\n"
                "Chapter " + df.get("chapter", "Unknown").astype(str) + ", Verse " + df.get("verse", "Unknown").astype(str)
            )
        else:
            df["formatted_response"] = (
                "Sanskrit: " + df.get("sanskrit", "N/A") + "\n"
                "Translation: " + df.get("translation", "N/A") + "\n"
                "Chapter " + df.get("chapter", "Unknown").astype(str) + ", Verse " + df.get("verse", "Unknown").astype(str)
            )

        responses = df["formatted_response"].tolist()
        combined_questions.extend(questions)
        combined_responses.extend(responses)

    return combined_questions, combined_responses

# Load datasets
file_paths = [
    "Bhagwad_Gita_Verses_English_Questions.csv",
    "Patanjali_Yoga_Sutras_Verses_English_Questions.csv"
]

questions, responses = load_and_combine_datasets(file_paths)
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
question_embeddings = embedding_model.encode(questions)

dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Function to retrieve relevant context
def retrieve_relevant_context(user_input):
    query_embedding = embedding_model.encode([user_input])
    _, indices = index.search(query_embedding, k=1)
    return responses[indices[0][0]]

# Check if BERT answer is valid
def is_valid_answer(answer):
    return len(answer.strip()) > 0 and answer.lower() not in ["", "n/a", "unknown", "[cls]"]

# Extract structured fields from retrieved context
def parse_context(context):
    fields = {
        "Speaker": "N/A",
        "Sanskrit": "N/A",
        "Translation": "N/A",
        "Chapter_Verse": "N/A"
    }
    # Regex to extract fields
    speaker_match = re.search(r"Speaker: (.+)", context,)
    sanskrit_match = re.search(r"Sanskrit: (.+)", context)
    translation_match = re.search(r"Translation: (.+)", context)
    chapter_verse_match = re.search(r"Chapter (\d+), Verse (\d+)", context)

    if speaker_match:
        fields["Speaker"] = speaker_match.group(1)
    if sanskrit_match:
        fields["Sanskrit"] = sanskrit_match.group(1)
    if translation_match:
        fields["Translation"] = translation_match.group(1)
    if chapter_verse_match:
        fields["Chapter_Verse"] = f"Chapter {chapter_verse_match.group(1)}, Verse {chapter_verse_match.group(2)}"

    return fields

# Generate response
def generate_response(user_input):
    start_time = time()
    retrieved_context = retrieve_relevant_context(user_input)
    parsed_context = parse_context(retrieved_context)
    
    inputs = bert_tokenizer(
        user_input,
        retrieved_context,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    outputs = bert_model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    summary = bert_tokenizer.convert_tokens_to_string(
        bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    ).strip()
    
    if is_valid_answer(summary):
        parsed_context["summary"] = summary
    else:
        parsed_context["summary"] = "No valid answer found"

    response_time = time() - start_time
    parsed_context["Response_Time"] = f"{response_time:.2f} seconds"

    # Manually format the response to add a newline after each field
    response = json.dumps(parsed_context, ensure_ascii=False, indent=2)
    
    # Add a newline after each response field
    response = re.sub(r'(?<=\n)', '\n', response)

    return response

# Chat function
def chat():
    print("Chatbot: Hi! I'm a chatbot using knowledge from the Bhagavad Gita and Yoga Sutras.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["hi"]:
            print("Chatbot: Hi! How can I help you?")
        elif user_input.lower() in ["bye", "goodbye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        else:
            response = generate_response(user_input)
            print("Chatbot:", response)

# Start the chat
chat()
