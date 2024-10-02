import torch
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

class BertForRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.r1_trait1_head = nn.Linear(config.hidden_size, 1)
        self.r1_trait2_head = nn.Linear(config.hidden_size, 1)
        self.r1_trait3_head = nn.Linear(config.hidden_size, 1)
        self.r1_trait4_head = nn.Linear(config.hidden_size, 1)

        self.r2_trait1_head = nn.Linear(config.hidden_size, 1)
        self.r2_trait2_head = nn.Linear(config.hidden_size, 1)
        self.r2_trait3_head = nn.Linear(config.hidden_size, 1)
        self.r2_trait4_head = nn.Linear(config.hidden_size, 1)
        self.overall_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        r1_trait1 = self.r1_trait1_head(cls_output)
        r1_trait2 = self.r1_trait2_head(cls_output)
        r1_trait3 = self.r1_trait3_head(cls_output)
        r1_trait4 = self.r1_trait4_head(cls_output)

        r2_trait1 = self.r2_trait1_head(cls_output)
        r2_trait2 = self.r2_trait2_head(cls_output)
        r2_trait3 = self.r2_trait3_head(cls_output)
        r2_trait4 = self.r2_trait4_head(cls_output)
        overall = self.overall_head(cls_output)
        return r1_trait1, r1_trait2, r1_trait3, r1_trait4, r2_trait1, r2_trait2, r2_trait3, r2_trait4, overall
    
class BertForRegressionForAllSets(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        input_dim = config.hidden_size + 15
        self.overall_head = nn.Linear(input_dim, 1)

    def forward(self, input_ids, attention_mask=None, features = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Concatenate features to the CLS output
        if features is not None:
            # Ensure features are of the right type and on the correct device
            features = features.to(cls_output.device, dtype=cls_output.dtype)
            combined_output = torch.cat((cls_output, features), dim=1)  # Concatenate along the feature dimension
        else:
            combined_output = cls_output  # If no features are provided, use only the CLS output
        overall = self.overall_head(combined_output)
        return overall
    
class BertForRegression2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        input_dim = config.hidden_size + 9
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.overall_head = nn.Linear(input_dim, 1)

    def forward(self, input_ids, attention_mask=None, features = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Concatenate features to the CLS output
        if features is not None:
            # Ensure features are of the right type and on the correct device
            features = features.to(cls_output.device, dtype=cls_output.dtype)
            combined_output = torch.cat((cls_output, features), dim=1)  # Concatenate along the feature dimension
            combined_output = self.batch_norm(combined_output)
        else:
            combined_output = cls_output  # If no features are provided, use only the CLS output
        overall = self.overall_head(combined_output)
        return overall
    
class BertForRegression_12(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        input_dim = config.hidden_size + 9
        self.dropout = nn.Dropout(p=0.3) 
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        self.content_head = nn.Linear(input_dim, 1)
        self.org_head = nn.Linear(input_dim, 1)
        self.wordchoice_head = nn.Linear(input_dim, 1)
        self.fluency_head = nn.Linear(input_dim, 1)
        self.convention_head = nn.Linear(input_dim, 1)
        
    def forward(self, input_ids, attention_mask=None, features = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Concatenate features to the CLS output
        if features is not None:
            # Ensure features are of the right type and on the correct device
            features = features.to(cls_output.device, dtype=cls_output.dtype)
            combined_output = torch.cat((cls_output, features), dim=1)  # Concatenate along the feature dimension
            combined_output = self.batch_norm(combined_output)
        else:
            combined_output = cls_output  # If no features are provided, use only the CLS output
        score_1 = self.content_head(combined_output)
        score_2 = self.org_head(combined_output)
        score_3 = self.wordchoice_head(combined_output)
        score_4 = self.fluency_head(combined_output)
        score_5 = self.convention_head(combined_output)
        return score_1, score_2, score_3, score_4, score_5
    

class BertForRegression_Source_Dependent(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        input_dim = config.hidden_size + 9
        self.dropout = nn.Dropout(p=0.3) 
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.content_head = nn.Linear(input_dim, 1)
        self.prompt_adh = nn.Linear(input_dim, 1)
        self.language = nn.Linear(input_dim, 1)
        self.narrativity = nn.Linear(input_dim, 1)
        
    def forward(self, input_ids, attention_mask=None, features = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Concatenate features to the CLS output
        if features is not None:
            # Ensure features are of the right type and on the correct device
            features = features.to(cls_output.device, dtype=cls_output.dtype)
            combined_output = torch.cat((cls_output, features), dim=1)  # Concatenate along the feature dimension
            combined_output = self.batch_norm(combined_output)
        else:
            combined_output = cls_output  # If no features are provided, use only the CLS output
        score_1 = self.content_head(combined_output)
        score_2 = self.prompt_adh(combined_output)
        score_3 = self.language(combined_output)
        score_4 = self.narrativity(combined_output)
        return score_1, score_2, score_3, score_4