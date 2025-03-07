import torch
from transformers import BertModel, BertTokenizer
import pickle
import os

task_dir = "/home/i53/student/boli/mask_pretrain/data_process/sam2/segmentation/videos/robo_data"


# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

def get_task_list(task_dir):
    task_list = [task_name for task_name in os.listdir(task_dir)]
    return task_list

# input a text and give embedding back
def generate_bert_embedding(text, model, tokenizer):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # Get embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the [CLS] token embedding as the sentence embedding
    # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_size]
    embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token is at index 0
    return embedding.squeeze().numpy()


task_list = get_task_list(task_dir)

# Generate embeddings for each task
task_embeddings = {}
for task in task_list:
    true_task = task[:-5]
    embedding = generate_bert_embedding(true_task, model, tokenizer)
    task_embeddings[true_task] = embedding

print(task_embeddings)

with open("real_robot_task_embeddings.pkl", 'wb') as f:
    pickle.dump(task_embeddings, f)