import json
import os
import joblib
import torch
import yfinance
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt

from APImodel import Model

def get_path_model(ticker, start_date, end_date):
    return os.getenv('PATH_MODEL').format(ticker=ticker, start_date=start_date, end_date=end_date)

def get_path_scaler(ticker, start_date, end_date):
    return os.getenv('PATH_SCALER').format(ticker=ticker, start_date=start_date, end_date=end_date)

def carregar_dados(ticker:str, start_date:str, end_date:str):
    """Carrega os dados de um ticker específico entre as datas fornecidas."""
    ticker_data = yfinance.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Preparar dados
    prices = ticker_data['Close'].values
    dates = ticker_data.index
    
    return prices, dates

def normalizar_precos(prices):
    """Normaliza os preços usando Min-Max Scaling."""
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    return prices_scaled, scaler

def create_sequences(data, sequence_length=30):
    """Cria sequências de dados para treinar o LSTM"""
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

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

def consular_modelos()-> list[Model]:
    models_aux:list[Model] = json.load(open(os.getenv('PATH_MODEL_JSON'), 'r'))
    return models_aux

def consultar_modelo(model_id: str) -> Model:
    """Carrega o modelo treinado a partir do ID"""
    models = consular_modelos()
    for model in models:
        if model.id == model_id:
            file_path = model.file
            scaler_path = model.scaler_file
            if os.path.exists(file_path) and os.path.exists(scaler_path):
                loaded_model = Model()
                loaded_model.ticker = model.ticker
                loaded_model.start_date = model.start_date
                loaded_model.end_date = model.end_date
                loaded_model.file = file_path
                loaded_model.scaler_file = scaler_path
                return loaded_model
    return None

def carregar_modelo(model_id: str):
    """Carrega o modelo treinado a partir do ID"""
    model = consultar_modelo(model_id)
    if model is None:
        raise ValueError(f"Modelo com ID {model_id} não encontrado.")
    
    file_path = model.file
    scaler_path = model.scaler_file
    ticker = model.ticker
    
    if not os.path.exists(file_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Modelo ou scaler não encontrados nos caminhos especificados.")
    
    # Carregar o modelo
    loaded_model = torch.load(file_path)
    
    # Carregar o scaler
    scaler = joblib.load(scaler_path)

    return loaded_model, scaler, ticker

def gravar_modelo(model, scaler, file_path, scaler_path)-> int:
    """Grava o modelo treinado em um arquivo"""
    torch.save(model.state_dict(), file_path)
    print(f"Modelo gravado em {file_path}")

    #Gravar o scaler em um arquivo
    joblib.dump(scaler, scaler_path)
    print(f"Scaler gravado em {scaler_path}")

    #Carregar o arquivo JSON em um dicionário
    id = hash(file_path)  # Gera um ID único baseado no caminho do arquivo
    
    #Incluir o novo modelo no dicionário de modelos
    models:list[Model] = consular_modelos()
    if not models:
        models = [] 
      
    model_aux: Model = model.model_dump()
    
    model_aux.id = str(id)
    model_aux.ticker = model.ticker
    model_aux.start_date = model.start_date
    model_aux.end_date = model.end_date
    model_aux.file = file_path
    model_aux.scaler_file = scaler_path
    
    models.append(model_aux)
    
    #Gravar o dicionário de modelos atualizado em um arquivo JSON, caso não exista, cria um novo
    with open(os.getenv('PATH_MODEL_JSON'), 'w') as f:
         json.dump(models, f, indent=4)
    
    print(f"Modelo ID: {id}")    
    return id
    
def plot_metrics_comparison(train_metrics, test_metrics):
    """Plota comparação das métricas entre treino, validação e teste"""
    metrics_names = list(train_metrics.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_names):
        values = [train_metrics[metric], test_metrics[metric]]
        labels = ['Treino', 'Teste']
        colors = ['#2E86AB', '#F18F01']
        
        bars = axes[i].bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        axes[i].set_title(f'{metric}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Valor')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Adiciona valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Comparação de Métricas - Treino vs Teste', 
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
    fig, (ax1) = plt.subplots(1, figsize=(15, 5))
    
    # Loss de treino
    if model.training_losses:
        ax1.plot(model.training_losses, color='#2E86AB', linewidth=2)
        ax1.set_title('Loss de Treinamento', fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('MSE Loss')
        ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()