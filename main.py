from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import yfinance
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class LightningLSTM(L.LightningModule):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.training_losses = []
        self.validation_losses = []
        
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
    
    def validation_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = nn.MSELoss()(output_i, label_i)
        self.log("val_loss", loss)
        self.validation_losses.append(loss.item())
        return loss

def create_sequences(data, sequence_length=30):
    """Cria sequências de dados para treinar o LSTM"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def split_data_temporal(data, dates, train_ratio=0.7, val_ratio=0.2):
    """Divide os dados temporalmente"""
    total_len = len(data)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_data = data[:train_len]
    train_dates = dates[:train_len]
    
    val_data = data[train_len:train_len + val_len]
    val_dates = dates[train_len:train_len + val_len]
    
    test_data = data[train_len + val_len:]
    test_dates = dates[train_len + val_len:]
    
    return (train_data, train_dates), (val_data, val_dates), (test_data, test_dates)

def calculate_mape(y_true, y_pred):
    """Calcula o Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_metrics(y_true, y_pred):
    """Calcula todas as métricas de avaliação"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }

def plot_metrics_comparison(train_metrics, val_metrics, test_metrics):
    """Plota comparação das métricas entre treino, validação e teste"""
    metrics_names = list(train_metrics.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_names):
        values = [train_metrics[metric], val_metrics[metric], test_metrics[metric]]
        labels = ['Treino', 'Validação', 'Teste']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        bars = axes[i].bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        axes[i].set_title(f'{metric}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Valor')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Adiciona valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Comparação de Métricas - Treino vs Validação vs Teste', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(dates, actual, predicted, title, subset_size=200):
    """Plota valores observados vs previstos"""
    # Se temos muitos dados, mostra apenas uma amostra para visualização
    if len(actual) > subset_size:
        step = len(actual) // subset_size
        dates_plot = dates[::step]
        actual_plot = actual[::step]
        predicted_plot = predicted[::step]
    else:
        dates_plot = dates
        actual_plot = actual
        predicted_plot = predicted
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(dates_plot, actual_plot, label='Valores Reais', 
             color='#2E86AB', linewidth=2, alpha=0.8)
    plt.plot(dates_plot, predicted_plot, label='Previsões', 
             color='#A23B72', linewidth=2, alpha=0.8)
    
    plt.title(f'{title}', fontsize=16, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_progress(model):
    """Plota o progresso do treinamento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss de treino
    if model.training_losses:
        ax1.plot(model.training_losses, color='#2E86AB', linewidth=2)
        ax1.set_title('Loss de Treinamento', fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('MSE Loss')
        ax1.grid(True, alpha=0.3)
    
    # Loss de validação
    if model.validation_losses:
        ax2.plot(model.validation_losses, color='#A23B72', linewidth=2)
        ax2.set_title('Loss de Validação', fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('MSE Loss')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def predict_sequences(model, sequences, scaler):
    """Faz previsões para um conjunto de sequências"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for seq in sequences:
            # Garantir que a sequência tenha a forma correta: (1, seq_len, 1)
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred = model(seq_tensor).item()
            predictions.append(pred)
    
    return np.array(predictions)

def main():
    print("📊 Iniciando análise LSTM com divisão temporal...")
    
    # Configurações
    SEQUENCE_LENGTH = 30
    TICKER = 'AAPL'
    START_DATE = '2015-01-01'
    END_DATE = '2024-12-31'
    
    # 1. CARREGAR DADOS
    print(f"📈 Baixando dados para {TICKER}...")
    ticker_data = yfinance.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    # Preparar dados
    prices = ticker_data['Close'].values
    dates = ticker_data.index
    
    print(f"✅ Dados carregados: {len(prices)} dias de {dates[0].strftime('%Y-%m-%d')} a {dates[-1].strftime('%Y-%m-%d')}")
    
    # 2. NORMALIZAÇÃO
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    # 3. CRIAR SEQUÊNCIAS
    X, y = create_sequences(prices_scaled, SEQUENCE_LENGTH)
    sequence_dates = dates[SEQUENCE_LENGTH:]
    
    print(f"📊 Sequências criadas: {len(X)} sequências de {SEQUENCE_LENGTH} dias")
    
    # 4. DIVISÃO TEMPORAL DOS DADOS
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.2)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    dates_train = sequence_dates[:train_size]
    dates_val = sequence_dates[train_size:train_size + val_size]
    dates_test = sequence_dates[train_size + val_size:]
    
    print(f"📋 Divisão dos dados:")
    print(f"  Treino: {len(X_train)} sequências ({dates_train[0].strftime('%Y-%m-%d')} a {dates_train[-1].strftime('%Y-%m-%d')})")
    print(f"  Validação: {len(X_val)} sequências ({dates_val[0].strftime('%Y-%m-%d')} a {dates_val[-1].strftime('%Y-%m-%d')})")
    print(f"  Teste: {len(X_test)} sequências ({dates_test[0].strftime('%Y-%m-%d')} a {dates_test[-1].strftime('%Y-%m-%d')})")
    
    # 5. CONVERTER PARA TENSORS
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # 6. CRIAR DATALOADERS
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 7. CRIAR E TREINAR MODELO
    print("🤖 Criando modelo LSTM...")
    model = LightningLSTM(input_size=1, hidden_size=64, output_size=1)
    
    # Treinar modelo
    trainer = L.Trainer(
        max_epochs=100,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir='./checkpoints'
    )
    
    print("🚀 Iniciando treinamento...")
    trainer.fit(model, train_loader, val_loader)
    
    # 8. FAZER PREVISÕES
    print("🔮 Fazendo previsões...")
    
    # Previsões para cada conjunto
    y_train_pred = predict_sequences(model, X_train, scaler)
    y_val_pred = predict_sequences(model, X_val, scaler)
    y_test_pred = predict_sequences(model, X_test, scaler)
    
    # Desnormalizar os dados
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_train_pred_rescaled = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    
    y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_val_pred_rescaled = scaler.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
    
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_rescaled = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    
    # 9. CALCULAR MÉTRICAS
    print("📊 Calculando métricas...")
    
    train_metrics = calculate_metrics(y_train_actual, y_train_pred_rescaled)
    val_metrics = calculate_metrics(y_val_actual, y_val_pred_rescaled)
    test_metrics = calculate_metrics(y_test_actual, y_test_pred_rescaled)
    
    # 10. MOSTRAR RESULTADOS
    print("\n" + "="*60)
    print("📈 RESULTADOS DAS MÉTRICAS")
    print("="*60)
    
    print(f"{'Conjunto':<12} {'MAE':<8} {'RMSE':<8} {'MAPE (%)':<10} {'R²':<8}")
    print("-" * 60)
    print(f"{'Treino':<12} {train_metrics['MAE']:<8.3f} {train_metrics['RMSE']:<8.3f} {train_metrics['MAPE']:<10.2f} {train_metrics['R²']:<8.3f}")
    print(f"{'Validação':<12} {val_metrics['MAE']:<8.3f} {val_metrics['RMSE']:<8.3f} {val_metrics['MAPE']:<10.2f} {val_metrics['R²']:<8.3f}")
    print(f"{'Teste':<12} {test_metrics['MAE']:<8.3f} {test_metrics['RMSE']:<8.3f} {test_metrics['MAPE']:<10.2f} {test_metrics['R²']:<8.3f}")
    print("="*60)
    
    # 11. GERAR GRÁFICOS
    print("📊 Gerando gráficos...")
    
    # Gráfico de métricas
    plot_metrics_comparison(train_metrics, val_metrics, test_metrics)
    
    # Gráfico de progresso do treinamento
    plot_training_progress(model)
    
    # Gráficos de previsões vs valores reais
    plot_predictions_vs_actual(dates_train, y_train_actual, y_train_pred_rescaled, 
                              f"Conjunto de Treino - {TICKER}")
    
    plot_predictions_vs_actual(dates_val, y_val_actual, y_val_pred_rescaled, 
                              f"Conjunto de Validação - {TICKER}")
    
    plot_predictions_vs_actual(dates_test, y_test_actual, y_test_pred_rescaled, 
                              f"Conjunto de Teste - {TICKER}")
    
    print("✅ Análise completa!")
    
    # Salvar modelo
    torch.save(model.state_dict(), f'lstm_model_{TICKER}.pth')
    print(f"💾 Modelo salvo como lstm_model_{TICKER}.pth")

if __name__ == "__main__":
    main()