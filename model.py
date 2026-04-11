from torch import nn
from transformers import BertModel
class BertClassifier(nn.Module):
    def __init__(self, config,num_classes):
        super().__init__()
        self.config=config
        self.num_classes=num_classes

        self.bert=BertModel.from_pretrained(self.config.model_path)
        self.Dropout=nn.Dropout(self.config.dropout)
        self.fc=nn.Linear(self.bert.config.hidden_size,self.num_classes)

    def forward(self,batch):
        outputs=self.bert(**batch)
        pooled_output=outputs.pooler_output
        x=self.Dropout(pooled_output)
        logits=self.fc(x)
        return logits
