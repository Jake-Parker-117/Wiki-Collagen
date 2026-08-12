import torch
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

CONFIG = {
    'model_name': 'facebook/esm2_t33_650M_UR50D',
    'num_labels': 11,
    'window_size': 1024,
    'step_size': 1024,
    'dropout': 0.35,
    'freeze_layers': 31
}

FAMILIES = ['Fibrillar', 'Network', 'FACIT', 'Multiplexin', 'MACIT', 'Misc. Collagen', 'Transmembrane CLP', 'C1q-containing CLP', 'C-type lectin-containing CLP', 'Ficolins', 'Misc. CLP']

# 3. Bidirectional Label Dictionary Translation Mappings
LABEL2ID = {family: idx for idx, family in enumerate(FAMILIES)}
ID2LABEL = {idx: family for family, idx in LABEL2ID.items()}

class CollagenFamilyClassifier(nn.Module):
    """
    ESM-2 encoder with a classification head for collagen family prediction.
    
    """

    def __init__(self, model_name, num_labels=CONFIG['num_labels'], dropout=0.35, freeze_layers=31):
        super().__init__()

        # Load ESM-2 and pre-trained weights from HuggingFace
        self.esm = EsmModel.from_pretrained(model_name)
        hidden_size = self.esm.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # Classification head: maps hidden_size → num_labels
        # Sets up three linear layers with normalisation (LayerNorm), Dropout (10%), and non-linearity (GELU) between them.
        linear1 = int(hidden_size * 0.8)
        linear2 = int(linear1 * 0.5) # if classification head is over or underfitting then adjust these values to retain or remove more data per layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, linear1),
            nn.LayerNorm(linear1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(linear1, linear2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(linear2, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Full forward pass
        
        Args:
            input_ids:      (batch, seq_len) integer token IDs
            attention_mask: (batch, seq_len) 1=real token, 0=padding
        """
        # Quicker mean pooling than a custom function
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        mask = attention_mask.unsqueeze(-1).float() 
        summed = torch.sum(token_embeddings * mask, dim=1) 
        counts = torch.sum(mask, dim=1).clamp(min=1e-9)
        pooled = summed / counts

        pooled = self.dropout(pooled) # dropout is bypassed
        return self.classifier(pooled)

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model = CollagenFamilyClassifier(model_name=CONFIG["model_name"], num_labels=CONFIG["num_labels"], dropout=CONFIG["dropout"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], use_fast=True)
    
    # Restore the best model weights from the saved checkpoint
    torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar]) # have to tell PyTorch to trust the numpy objects
    checkpoint = torch.load("ESM_2_CollagenAI.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(device) # load best model in full into gpu
    model.eval() # Put it in evaluation mode
    
    return model, tokenizer, device

def get_seqs(fasta_file, colAI_txt):
    header = None
    new_line = []
    sequence = ""
    with open(fasta_file, 'r', encoding='utf-8') as f:
        with open(colAI_txt, 'w', encoding='utf-8') as output:
            for line in f:
                if not line: #skip empty lines
                    continue
                if line.startswith(">"):
                    if header is not None:
                        sequence = "".join(new_line)
                        output.write(f"{header}\n{sequence}\n\n") #Saves the header and combined sequence if the pattern is found
                        
                        header = line #sets the new header
                        new_line = [] #resets the new line object for next sequence
                    else:
                        header = line #sets 1st header
                else:
                    new_line.append(line) #capturing sequence lines

            if header is not None: #capturing the last input
                sequence = "".join(new_line)
                output.write(f"{header}\n{sequence}\n\n")

def standard_predict(sequence, model, tokenizer, device):
    sequence = sequence.upper().strip().replace(" ", "")
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            tokeniser_output = tokenizer(sequence, return_tensors="pt", padding="do_not_pad", truncation=True, max_length=CONFIG['window_size']) 
            model_inputs = {k: v.to(device) for k, v in tokeniser_output.items()} # send tokenised sequences and attention mask to gpu
            logits = model(model_inputs['input_ids'], attention_mask=model_inputs['attention_mask']) # get logits from model
            
            predicted_class_id = logits.argmax(dim=-1) #predicts 1 class
            probs = torch.softmax(logits, dim=-1) # finds probability for each class
    
        prediction = predicted_class_id.cpu().item()
        confidence = probs[0][predicted_class_id].cpu().item()
    
    return ID2LABEL[prediction], confidence


def predict_long_sequence(sequence, model, tokenizer, device, window_size=CONFIG['window_size'], step=CONFIG['step_size'], batch_size=2):
    """
    same as standard but uses a sliding window to chop longer seqeunces into overlapping 1024-aa windows, gets predictions for each, and averages the scores.
    """
    sequence = sequence.upper().strip().replace(" ", "")
    chunks = []
    
    for i in range(0, len(sequence) - window_size + 1, step): # Creating a chunks list to be sent to the gpu once rather than over and over
        chunks.append(sequence[i:i+window_size])
    if not chunks or chunks[-1] != sequence[-window_size:]: # Catch trailing chunks
        chunks.append(sequence[-window_size:])
    
    all_probs = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i +batch_size]
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                tokeniser_output = tokenizer(batch_chunks, return_tensors="pt", padding="do_not_pad", max_length=window_size)
                
                model_inputs = {k: v.to(device) for k, v in tokeniser_output.items()}
                logits = model(model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'])
                
                probs = torch.softmax(logits, dim=-1) # Convert to probabilities

                all_probs.append(probs.cpu())

    final_probs = torch.mean(torch.cat(all_probs, dim=0), dim=0)
    predicted_class_id = torch.argmax(final_probs, dim=-1)
    
    prediction = predicted_class_id.item()
    confidence = final_probs[predicted_class_id].item()
    
    return ID2LABEL[prediction], confidence

def make_prediction(colAI_txt, classifications_txt, confidence_threshold, model, tokenizer, device, window_size): # Need to implement a checker on the confidence so it won't predict any old sequence
    
    confidence_threshold = confidence_threshold/100 if confidence_threshold > 1 else confidence_threshold # using decimal rather than percentage
    
    with open(colAI_txt, 'r', encoding='utf-8') as f:
        in_lines = f.readlines()
    out_lines = []
    
    for line in in_lines:
        line = line.strip()
        if not line: #skip empty lines
                continue
            
        if line.startswith(">"):
            header = line
            continue
        
        
        seq = line
        if len(seq) <= window_size:
            pred, conf = standard_predict(seq, model, tokenizer, device)
        else:
            pred, conf = predict_long_sequence(seq, model, tokenizer, device, window_size=CONFIG['window_size'], step=CONFIG['step_size'])
                    
        conf=f"{conf*100:.2f}% (Below Confidence Threshold)" if conf<confidence_threshold else f"{conf*100:.2f}%"
                    
        out_lines.append(f"{header}:\n Prediction: {pred} | Confidence: {conf}\n\n")
    
    with open(classifications_txt, 'w', encoding='utf-8') as o:
        o.writelines(out_lines)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache() # Flush GPU memory after writing sequences