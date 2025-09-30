import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not found. SBERT will not be functional.")

class SBERTTransformer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if SBERT_AVAILABLE: self.model = SentenceTransformer(model_name)
        else: self.model = None
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        if self.model and X is not None: return self.model.encode(list(X))
        return []

class MDASentimentModel:
    def __init__(self, model_path):
        self.model, self.tokenizer = None, None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            class BERTSentimentClassifier(nn.Module):
                def __init__(self, n_classes):
                    super(BERTSentimentClassifier, self).__init__()
                    self.bert = AutoModel.from_pretrained('bert-base-uncased')
                    self.drop = nn.Dropout(p=0.3)
                    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
                def forward(self, input_ids, attention_mask):
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                    output = self.drop(outputs[1])
                    return self.out(output)
            self.model = BERTSentimentClassifier(n_classes=5)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.bert.resize_token_embeddings(30873) # FIX for size mismatch
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            logger.info(f"MDA sentiment model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load MDA model from {model_path}: {e}")
    def is_available(self): return self.model is not None
    def predict(self, texts: list): return [], []