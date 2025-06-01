import torch
import pytorch_lightning as L
from torch.utils.data import TensorDataset, DataLoader
from util import metrics, util
from model import lstm
import numpy as np

def __separate_train_test_data__(x, y, dates, train_size=0.8):
    """Divide os dados em conjuntos de treino e teste com base na proporção fornecida"""
    train_len = int(len(x) * train_size)
    
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    
    dates_train = dates[:train_len]
    dates_test = dates[train_len:]
    return x_train, y_train, x_test, y_test, dates_train, dates_test


def __predict_sequences__(model, sequences, scaler):
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

def train_model(ticker: str, start_date: str, end_date: str, train_size: float, sequence_length: int, num_epochs: int = 10) -> dict:
    prices = []
    dates = []
    
    # 1. CARREGAR DADOS
    print(f"Baixando dados para {ticker}...")
    prices, dates = util.carregar_dados(ticker, start_date, end_date)
    print(f"Dados carregados: {len(prices)} dias de {dates[0].strftime('%Y-%m-%d')} a {dates[-1].strftime('%Y-%m-%d')}")
    
    if len(prices) == 0:
        raise ValueError("Nenhum dado de preço encontrado para o ticker especificado.")

    # 2. NORMALIZAÇÃO
    print("Normalizando os preços...")
    prices_scaled, scaler = util.normalizar_precos(prices)
    
    # 3. CRIAR SEQUÊNCIAS
    x, y = util.create_sequences(prices_scaled, sequence_length)
    sequence_dates = dates[sequence_length:]
    print(f"Sequências criadas: {len(x)} sequências de {sequence_length} dias")

    # 4. DIVISÃO TEMPORAL DOS DADOS
    x_train, y_train, x_test, y_test, dates_train, dates_test = __separate_train_test_data__(x, y, sequence_dates, train_size=train_size)
    print("Divisão dos dados:")
    print(f"  Treino: {len(x_train)} sequências ({dates_train[0].strftime('%Y-%m-%d')} a {dates_train[-1].strftime('%Y-%m-%d')})")
    print(f"  Teste: {len(x_test)} sequências ({dates_test[0].strftime('%Y-%m-%d')} a {dates_test[-1].strftime('%Y-%m-%d')})")

    # 5. CONVERTER PARA TENSORS
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # 6. CRIAR DATALOADERS
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 7. CRIAR E TREINAR MODELO
    print("Criando modelo LSTM...")
    model = lstm.LightningLSTM(input_size=1, hidden_size=64, output_size=1)
    
    # Treinar modelo
    trainer = L.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=10,
        enable_progress_bar=False,
        logger=False, 
        enable_checkpointing=False
    )
    
    print("Iniciando treinamento...")
    trainer.fit(model, train_loader)
    
    # 8. FAZER PREVISÕES
    print("Fazendo previsões...")
    
    # Previsões para cada conjunto
    #y_train_pred = lstm.predict_sequences(model, x_train, scaler)
    y_test_pred = __predict_sequences__(model, x_test, scaler)
    
    # Desnormalizar os dados
    #y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    #y_train_pred_rescaled = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_rescaled = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    
    # 9. CALCULAR MÉTRICAS
    print("Calculando métricas...")
    
    #train_metrics = metrics.calculate_metrics(y_train_actual, y_train_pred_rescaled)
    test_metrics = metrics.calculate_metrics(y_test_actual, y_test_pred_rescaled)
    
    #10. SALVAR MODELO
    print("Salvando modelo...")
    test_metrics["ID"] = util.gravar_modelo(model, scaler, ticker, start_date, end_date, util.get_path_model(ticker, start_date, end_date), util.get_path_scaler(ticker, start_date, end_date))
    print(f"Modelo salvo com ID: {test_metrics['ID']}")

    return test_metrics

def predict_model(id, start_date, end_date, num_days_to_predict=1, sequence_length=30):
    """
    Faz previsões para um número especificado de dias futuros usando o modelo treinado
    
    Args:
        id: ID do modelo salvo
        initial_sequence: Sequência inicial de preços (deve ter pelo menos sequence_length valores)
        num_days_to_predict: Número de dias futuros para prever
        sequence_length: Comprimento da sequência usada pelo modelo (padrão: 30)

    Returns:
        numpy.array: Array com as previsões para os próximos num_days_to_predict dias
    """
    # Carregar modelo e scaler
    model = lstm.LightningLSTM(input_size=1, hidden_size=64, output_size=1)
    model_state, scaler, ticker = util.carregar_modelo(id)
    model.load_state_dict(model_state)
    
    if model is None or scaler is None:
        raise ValueError("Modelo ou scaler não encontrado para o ID fornecido.")
    model.eval()

    # Carregar dados para previsão
    initial_sequence, _ = util.carregar_dados(ticker, start_date, end_date)

    if model is None or scaler is None:
        raise ValueError("Modelo ou scaler não encontrado para o ID fornecido.")
    
    if len(initial_sequence) < sequence_length:
        raise ValueError(f"A sequência inicial deve ter pelo menos {sequence_length} valores.")
    
    model.eval()
    predictions = []
    
    # Normalizar a sequência inicial
    initial_sequence_normalized = scaler.transform(np.array(initial_sequence).reshape(-1, 1)).flatten()
    
    # Usar os últimos sequence_length valores como ponto de partida
    current_sequence = initial_sequence_normalized[-sequence_length:].copy()
    
    with torch.no_grad():
        for _ in range(num_days_to_predict):
            # Preparar a sequência para o modelo: (1, seq_len, 1)
            seq_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # Fazer a previsão (valor normalizado)
            pred_normalized = model(seq_tensor).item()
            
            # Desnormalizar a previsão
            pred_actual = scaler.inverse_transform([[pred_normalized]])[0][0]
            predictions.append(pred_actual)
            
            # Atualizar a sequência: remover primeiro valor e adicionar a previsão normalizada
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_normalized
    
    return predictions