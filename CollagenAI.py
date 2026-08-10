import os
import torch
import torch.nn as nn
import numpy as np
from transformers import EsmTokenizer, EsmModel

CONFIG = {
    'model_name': 'facebook/esm2_t33_650M_UR50D',
    'num_labels': 11,
    'window_size': 1024,
    'checkpoint_path': '/collagen-project/checkpoints/best_collagen_model.pt',
    'dropout': 0.35,
    'freeze_layers': 31
    }

FAMILIES = ['fibrillar', 'network', 'facit', 'multiplexin', 'macit', 'other', 'transmembrane_clp', 'c1q_containing_clp', 'c_type_lectin_clp', 'ficolin_clp', 'miscellaneous_clp']
PLOT_FAMILIES = ['Fibrillar', 'Network', 'FACIT', 'Multiplexin', 'MACIT', 'Misc. Collagen', 'Transmembrane CLP', 'C1q-containing CLP', 'C-type lectin-containing CLP', 'Ficolins', 'Misc. CLP']
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

        # Freeze embedding layer and number of transformer layers specified
        self._freeze_layers(freeze_layers)

        # Count and display parameter statistics
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        # Setting dropout
        self.dropout = nn.Dropout(dropout)

        # Classification head: maps hidden_size → num_labels
        # Sets up three linear layers with normalisation (LayerNorm), Dropout (10%), and non-linearity (GELU) between them.
        linear1 = int(hidden_size * 0.8)
        linear2 = int(linear1 * 0.5) # if classification head is over or underfitting then adjust these values to retain or remove more data per layer
        print(f"Classification head: {hidden_size} → {linear1} → {linear2} → {num_labels} \n Adjust data retention if over/underfitting")
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

    def _freeze_layers(self, num_freeze):
        """
        Freeze the embedding layer and specified number of initial transformer layers.
        """
        # Freeze the embedding layer (converts token IDs to initial vectors)
        for param in self.esm.embeddings.parameters():
            param.requires_grad = False

        # Freeze the first num_freeze transformer layers
        for layer in self.esm.encoder.layer[:num_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    def mean_pool(self, token_embeddings, attention_mask):
        """
        Average embeddings across the sequence length dimension.
        Uses attention mask to exclude padding tokens

        Args:
            token_embeddings: (batch, seq_len, hidden_size)
            attention_mask:   (batch, seq_len) — 1 for real tokens, 0 for padding
        """
        # Expand mask to match embedding dimensions for element-wise multiplication
        mask   = attention_mask.unsqueeze(-1).float() # (batch, seq_len, 1)
        summed = (token_embeddings * mask).sum(dim=1) # (batch, hidden_size)
        counts = mask.sum(dim=1).clamp(min=1e-9) # (batch, 1) avoid div by zero
        return summed / counts # (batch, hidden_size)

    def forward(self, input_ids, attention_mask):
        """
        Full forward pass
        
        Args:
            input_ids:      (batch, seq_len) integer token IDs
            attention_mask: (batch, seq_len) 1=real token, 0=padding
        """
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        token_embeddings = outputs.last_hidden_state #takes contextual embedding for each token position (ie, (batch, seq len, hidden size))
        pooled = self.mean_pool(token_embeddings, attention_mask) # mean pooling (ie, averaging across sequence positions)

        pooled = self.dropout(pooled) # apply dropout
        logits = self.classifier(pooled) # use classification head to get logits

        return logits

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model = CollagenFamilyClassifier(model_name=CONFIG["model_name"], num_labels=CONFIG["num_labels"], dropout=CONFIG["dropout"])
    tokenizer = EsmTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Restore the best model weights from the saved checkpoint
    torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar]) # have to tell PyTorch to trust the numpy objects
    checkpoint = torch.load("CollagenAI/best_collagen_model_eleven_class.pt", map_location=device, weights_only=False)
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
        tokeniser_output = tokenizer(sequence, return_tensors="pt", padding="do_not_pad", truncation=True, max_length=CONFIG['window_size']) 
        model_inputs = {k: v.to(device) for k, v in tokeniser_output.items()} # send tokenised sequences and attention mask to gpu
        logits = model(model_inputs['input_ids'], attention_mask=model_inputs['attention_mask']) # get logits from model
        
        predicted_class_id = logits.argmax(dim=-1).cpu().numpy() #predicts 1 class
        probs = nn.functional.softmax(logits, dim=-1) # finds probability for each class
    
        prediction = ID2LABEL[predicted_class_id.item()]
        confidence = probs[0][predicted_class_id].item()
    
    return prediction, confidence


def predict_long_sequence(sequence, model, tokenizer, device, window_size=CONFIG['window_size'], step=256):
    """
    same as standard but uses a sliding window to chop longer seqeunces into overlapping 1024-aa windows, gets predictions for each, and averages the scores.
    """
    sequence = sequence.upper().strip().replace(" ", "")
    all_logits = []
    for i in range(0, len(sequence) - window_size + 1, step): # sliding window with increments of 256 (test and see how this performs on the test dataset)
        chunk = sequence[i:i+window_size]
        with torch.no_grad():
            tokeniser_output = tokenizer(chunk, return_tensors="pt", padding="do_not_pad", max_length=window_size)
            model_inputs = {k: v.to(device) for k, v in tokeniser_output.items()}
            logits = model(model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'])
            probs = nn.functional.softmax(logits, dim=-1) # Convert to probabilities
            all_logits.append(probs)
            
    # Pool the probabilities across the entire length of the protein
    final_probs = torch.mean(torch.stack(all_logits), dim=0)
    predicted_class_id = torch.argmax(final_probs, dim=-1).item()
    
    prediction = ID2LABEL[predicted_class_id]
    confidence = final_probs[0][predicted_class_id].item()
    
    return prediction, confidence

def make_prediction(colAI_txt, classifications_txt, confidence_threshold, model, tokenizer, device, window_size): # Need to implement a checker on the confidence so it won't predict any old sequence
    with open(colAI_txt, 'r', encoding='utf-8') as f:
        with open(classifications_txt, 'w', encoding='utf-8') as o:
            for line in f:
                if not line: #skip empty lines
                    continue
                if line.startswith(">"):
                    header = line
                    continue
                else:
                    seq = line
                    if len(seq) <= window_size:
                        pred, conf = standard_predict(seq, model, tokenizer, device)
                    else:
                        pred, conf = predict_long_sequence(seq, model, tokenizer, device, window_size=1024, step=256)
                    
                    if conf<confidence_threshold:
                        conf="Below Confidence Threshold"
                    
                    o.write(f"{header}:\n Prediction:{pred} Confidence: {conf}\n\n")