import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel

import config


class DistilBert(nn.Module):
    def __init__(self, pretrained_model_name=config.BERT_MODEL_NAME, num_classes=2):
        super().__init__()
        model_config = AutoConfig.from_pretrained(pretrained_model_name)

        self.distilbert = AutoModel.from_pretrained(pretrained_model_name,
                                                    config=model_config)
        self.pre_classifier = nn.Linear(model_config.dim, model_config.dim)
        self.dropout = nn.Dropout(model_config.seq_classif_dropout)
        self.out = nn.Linear(model_config.dim, num_classes)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        assert attention_mask is not None

        model_output = self.distilbert(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       head_mask=head_mask)
        # [BATCH_SIZE=BS, MAX_SEQ_LENGTH = 512, DIM = 768]
        hidden_state = model_output[0]
        # get the first token as pooled_output
        pooled_output = hidden_state[:, 0]  # [BS, 768]
        pooled_output = self.pre_classifier(pooled_output)  # [BS, 768]
        pooled_output = F.relu(pooled_output)  # [BS, 768]
        pooled_output = self.dropout(pooled_output)  # [BS, 768]
        output = self.out(pooled_output)  # [BS, 2]

        return output


class Model:
    def __init__(self):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)

        classifier = DistilBert(config.BERT_MODEL_NAME,
                                len(config.CLASS_NAMES))
        classifier.load_state_dict(
            torch.load(config.TRAINED_MODEL_PATH, map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer(
            text,
            max_length=config.MAX_SEQUENCE_LEN,
            padding='max_length',
            return_tensors="pt"
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(
                input_ids, attention_mask), dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()

        sentiment = config.CLASS_NAMES[predicted_class]
        sentiment_proba_dict = dict(zip(config.CLASS_NAMES,
                                        probabilities))

        return (sentiment, confidence, sentiment_proba_dict)


model = Model()


def get_model():
    return model
