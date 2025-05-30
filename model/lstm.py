import torch.nn as nn
import pytorch_lightning as L
from torch.optim import Adam

class LightningLSTM(L.LightningModule):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.training_losses = []
        
    def forward(self, input, lengths=None):
        # Garantir que o input tenha a forma correta: (batch_size, seq_len, input_size)
        if len(input.shape) == 2:
            # Se for (batch_size, seq_len), adicionar dimensão de feature
            input = input.unsqueeze(-1)
        elif len(input.shape) == 1:
            # Se for apenas uma sequência, adicionar batch e feature dimensions
            input = input.unsqueeze(0).unsqueeze(-1)
            
        lstm_out, (hidden, cell) = self.lstm(input)
        
        prediction = self.fc(hidden[-1])
        return prediction.squeeze()
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
        
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = nn.MSELoss()(output_i, label_i)
        self.log("train_loss", loss)
        self.training_losses.append(loss.item())
        return loss