import io
import json
#import os
import joblib
import torch
import yfinance
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from util import aws
from model.APImodel import Model

PATH_MODEL_JSON = "models.json"
PATH_MODEL = "modelos_treinados"

def get_path_model(ticker, start_date, end_date):
    #os.makedirs(os.getenv('PATH_MODEL'), exist_ok=True)
    #return os.path.join(os.getcwd(), os.getenv('PATH_MODEL'), f"{ticker}_{start_date}_{end_date}_lstm.pth")
    return f"{PATH_MODEL}/{ticker}_{start_date}_{end_date}_lstm.pth"

def get_path_scaler(ticker, start_date, end_date):
    #os.makedirs(os.getenv('PATH_MODEL'), exist_ok=True)
    #return os.path.join(os.getcwd(), os.getenv('PATH_MODEL'), f"{ticker}_{start_date}_{end_date}_scaler.save")
    return f"{PATH_MODEL}/{ticker}_{start_date}_{end_date}_scaler.save"

def get_path_json():
    #return os.path.join(os.getcwd(), os.getenv('PATH_MODEL'), os.getenv('PATH_MODEL_JSON'))
    return f"{PATH_MODEL}/{PATH_MODEL_JSON}"

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
        #with open(get_path_json(), 'r', encoding='utf-8') as file_data:
        with aws.carregar_json_s3(get_path_json()) as file_data:
            data = json.load(file_data)
        # Convert each dict to Model object
        return [Model.model_validate(item) for item in data]
    except Exception:
        return []

def consultar_modelo(model_id: str) -> Model:
    """Carrega o modelo treinado a partir do ID"""
    models = consultar_modelos()
    for model in models:
        if model.id == model_id:
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
    
    #if not os.path.exists(file_path) or not os.path.exists(scaler_path):
    #    raise FileNotFoundError("Modelo ou scaler não encontrados nos caminhos especificados.")

    model_buffer, scaler_buffer = aws.carregar_lstm_s3(file_path, scaler_path)

    # Carregar o modelo
    loaded_model_state = torch.load(model_buffer)
    
    # Carregar o scaler
    scaler = joblib.load(scaler_buffer)

    return loaded_model_state, scaler, ticker

def gravar_modelo(lstm_model, scaler, ticker, start_date, end_date, file_path, scaler_path)-> int:
    """Grava o modelo treinado em um arquivo"""
    #torch.save(lstm_model.state_dict(), file_path)
    #print(f"Modelo gravado em {file_path}")

    #Gravar o scaler em um arquivo
    #joblib.dump(scaler, scaler_path)
    #print(f"Scaler gravado em {scaler_path}")

    # Gravar o state dict do modelo LSTM no S3
    buffer_model = io.BytesIO()
    torch.save(lstm_model.state_dict(), buffer_model)
    buffer_model.seek(0)  # Reset buffer position
    aws.gravar_s3(buffer_model, file_path)
    print(f"Modelo gravado em {file_path}")
    
    # Gravar o scaler no S3
    buffer_scaler = io.BytesIO()
    joblib.dump(scaler, buffer_scaler)
    buffer_scaler.seek(0)  # Reset buffer position
    aws.gravar_s3(buffer_scaler, scaler_path)
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
    file_path = get_path_model(ticker, start_date, end_date)
    scaler_path = get_path_scaler(ticker, start_date, end_date)
    
    model_aux = Model(id=id, ticker=ticker, start_date=start_date, end_date=end_date, file=file_path, scaler_file=scaler_path)
    modelos.append(model_aux)

    #Gravar o dicionário de modelos atualizado em um arquivo JSON, caso não exista, cria um novo
    #with open(get_path_json(), 'w') as f:
    #    f.write(json.dumps([modelo.model_dump() for modelo in modelos], indent=4, ensure_ascii=False))
    aws.gravar_s3(json.dumps([modelo.model_dump() for modelo in modelos], indent=4, ensure_ascii=False), get_path_json())
    print(f"Modelo ID: {id}")
    return id