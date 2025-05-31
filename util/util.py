import json
import os
import joblib
import torch
import yfinance
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from model.APImodel import Model

def get_path_model(ticker, start_date, end_date):
    os.makedirs(os.getenv('PATH_MODEL'), exist_ok=True)
    return os.path.join(os.getcwd(), os.getenv('PATH_MODEL'), f"{ticker}_{start_date}_{end_date}_lstm.pth")

def get_path_scaler(ticker, start_date, end_date):
    os.makedirs(os.getenv('PATH_MODEL'), exist_ok=True)
    return os.path.join(os.getcwd(), os.getenv('PATH_MODEL'), f"{ticker}_{start_date}_{end_date}_scaler.save")

def get_path_json():
    return os.path.join(os.getcwd(), os.getenv('PATH_MODEL'), os.getenv('PATH_MODEL_JSON'))

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

def consultar_modelos() -> list[Model]:
    try:
        with open(get_path_json(), 'r', encoding='utf-8') as file:
            data = json.load(file)
        # Convert each dict to Model object
        return [Model.model_validate(item) for item in data]
    except Exception:
        return []

def consultar_modelo(model_id: str) -> Model:
    """Carrega o modelo treinado a partir do ID"""
    models = consultar_modelos()
    for model in models:
        if model.id == model_id:
            file_path = model.file
            scaler_path = model.scaler_file
            if os.path.exists(file_path) and os.path.exists(scaler_path):
                return model
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
    loaded_model_state = torch.load(file_path)
    
    # Carregar o scaler
    scaler = joblib.load(scaler_path)

    return loaded_model_state, scaler, ticker

def gravar_modelo(lstm_model, scaler, ticker, start_date, end_date, file_path, scaler_path)-> int:
    """Grava o modelo treinado em um arquivo"""
    torch.save(lstm_model.state_dict(), file_path)
    print(f"Modelo gravado em {file_path}")

    #Gravar o scaler em um arquivo
    joblib.dump(scaler, scaler_path)
    print(f"Scaler gravado em {scaler_path}")

    #Incluir o novo modelo no dicionário de modelos
    modelos:list[Model] = consultar_modelos()
    if not modelos:
        modelos = [] 
        id = 1
    else:
        if any(modelo.ticker == ticker and modelo.start_date == start_date and modelo.end_date == end_date for modelo in modelos):
            id = (modelo.id for modelo in modelos if modelo.ticker == ticker and modelo.start_date == start_date and modelo.end_date == end_date)
        else:
            id = max(modelo.id for modelo in modelos) + 1
        
    #Criar um novo modelo
    file_path = os.path.abspath(file_path)
    scaler_path = os.path.abspath(scaler_path)
    
    model_aux = Model(id=id, ticker=ticker, start_date=start_date, end_date=end_date, file=file_path, scaler_file=scaler_path)
    modelos.append(model_aux)

    #Gravar o dicionário de modelos atualizado em um arquivo JSON, caso não exista, cria um novo
    with open(get_path_json(), 'w') as f:
         f.write(json.dumps([modelo.model_dump() for modelo in modelos], indent=4, ensure_ascii=False))

    print(f"Modelo ID: {id}")
    return id