from imports import *

class lm_classifier(nn.Module):
    def __init__(self,bert, dropout):

        super(lm_classifier, self).__init__()
        self.bert = bert.to(device)
        
        D_in, H1, H2, D_out = 768, 256, 64, 3
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Linear(H2, D_out)
        )

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, input_ids, attention_mask):
      outputs = self.bert(input_ids, attention_mask)
      logits = self.classifier(outputs[0][:,0,:])

      return logits
  
class CNNBert(nn.Module):
    
    def __init__(self, bert, dropout):
        super(CNNBert, self).__init__()
        self.bert = bert.to(device)
        
        filter_sizes = [1,2,3,4,5]
        D_in = 768
        N_filter = 32
        
        self.convs1 = nn.ModuleList([nn.Conv2d(2, N_filter, (K, D_in)) for K in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes)*N_filter, 3)
        self.sigmoid = nn.Sigmoid()
        self.bert_model = bert
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        x = self.bert_model(input_ids, attention_mask=attention_mask)[2][-2:]
        x = torch.stack(x, dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x)
        return logit