import json
from fastapi.responses import RedirectResponse
import torch
import pytorch_lightning as L
from torch.utils.data import TensorDataset, DataLoader
from repository import lstmRepository
from util import metrics, plot, util, validate
from model import lstm
from model.APImodel import Model, TrainRequest, PredictionRequest
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Tech Challenge 04")

@app.get("/", include_in_schema=False)
async def docs_redirect():
    """
    Redireciona para a documentação Swagger.
    """
    return RedirectResponse(url="/docs")

@app.get("/models")
async def get_models() -> list[Model]:
    models: list[Model] = util.consultar_modelos()
    return models

@app.post("/models/train")
async def train_model(train_request: TrainRequest):
    try:
        validate.validate_train_request(train_request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        training_metrics = lstmRepository.train_model(
            ticker=train_request.ticker,
            start_date=train_request.start_date,
            end_date=train_request.end_date,
            train_size=train_request.train_size,
            sequence_length=train_request.sequence_length
        )

        return {"model_id": training_metrics["ID"],
                "ticker": train_request.ticker,
                "mae"  : training_metrics['MAE'],
                "rmse" : training_metrics['RMSE'],
                "mape" : training_metrics['MAPE'],
                "r2"   : training_metrics['R²']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/models/predict")
async def predict_model(predict_request: PredictionRequest):
    try:
        validate.validate_prediction_request(predict_request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))    

    try:
        predictions = lstmRepository.predict_model(
            predict_request.model_id,
            predict_request.start_date,
            predict_request.end_date,
            predict_request.days,
            predict_request.sequence_length
        )
        return json.dumps(predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    print("📊 Iniciando análise LSTM com divisão temporal...")
    
    # Configurações
    SEQUENCE_LENGTH = 30
    TICKER = 'AAPL'
    START_DATE = '2015-01-01'
    END_DATE = '2024-12-31'

    prices = []
    dates = []
    
    # 1. CARREGAR DADOS
    print(f"📈 Baixando dados para {TICKER}...")
    prices, dates = util.carregar_dados(TICKER, START_DATE, END_DATE)
    print(f"✅ Dados carregados: {len(prices)} dias de {dates[0].strftime('%Y-%m-%d')} a {dates[-1].strftime('%Y-%m-%d')}")

    # 2. NORMALIZAÇÃO
    print("🔄 Normalizando os preços...")
    prices_scaled, scaler = util.normalizar_precos(prices)
    
    # 3. CRIAR SEQUÊNCIAS
    x, y = util.create_sequences(prices_scaled, SEQUENCE_LENGTH)
    sequence_dates = dates[SEQUENCE_LENGTH:]
    print(f"📊 Sequências criadas: {len(x)} sequências de {SEQUENCE_LENGTH} dias")

    # 4. DIVISÃO TEMPORAL DOS DADOS
    x_train, y_train, x_test, y_test, dates_train, dates_test = lstm.separate_train_test_data(x, y, sequence_dates, train_size=0.8)
    print("📋 Divisão dos dados:")
    print(f"  Treino: {len(x_train)} sequências ({dates_train[0].strftime('%Y-%m-%d')} a {dates_train[-1].strftime('%Y-%m-%d')})")
    print(f"  Teste: {len(x_test)} sequências ({dates_test[0].strftime('%Y-%m-%d')} a {dates_test[-1].strftime('%Y-%m-%d')})")

    # 5. CONVERTER PARA TENSORS
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # 6. CRIAR DATALOADERS
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 7. CRIAR E TREINAR MODELO
    print("🤖 Criando modelo LSTM...")
    model = lstm.LightningLSTM(input_size=1, hidden_size=64, output_size=1)
    
    # Treinar modelo
    trainer = L.Trainer(
        max_epochs=100,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir='./checkpoints'
    )
    
    print("🚀 Iniciando treinamento...")
    trainer.fit(model, train_loader)
    
    # 8. FAZER PREVISÕES
    print("🔮 Fazendo previsões...")
    
    # Previsões para cada conjunto
    y_train_pred = lstm.predict_sequences(model, x_train, scaler)
    y_test_pred = lstm.predict_sequences(model, x_test, scaler)
    
    # Desnormalizar os dados
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_train_pred_rescaled = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_rescaled = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    
    # 9. CALCULAR MÉTRICAS
    print("📊 Calculando métricas...")
    
    train_metrics = metrics.calculate_metrics(y_train_actual, y_train_pred_rescaled)
    test_metrics = metrics.calculate_metrics(y_test_actual, y_test_pred_rescaled)
    
    # 10. MOSTRAR RESULTADOS
    print("\n" + "="*60)
    print("📈 RESULTADOS DAS MÉTRICAS")
    print("="*60)
    
    print(f"{'Conjunto':<12} {'MAE':<8} {'RMSE':<8} {'MAPE (%)':<10} {'R²':<8}")
    print("-" * 60)
    print(f"{'Treino':<12} {train_metrics['MAE']:<8.3f} {train_metrics['RMSE']:<8.3f} {train_metrics['MAPE']:<10.2f} {train_metrics['R²']:<8.3f}")
    print(f"{'Teste':<12} {test_metrics['MAE']:<8.3f} {test_metrics['RMSE']:<8.3f} {test_metrics['MAPE']:<10.2f} {test_metrics['R²']:<8.3f}")
    print("="*60)
    
    # 11. GERAR GRÁFICOS
    print("📊 Gerando gráficos...")
    
    # Gráfico de métricas
    plot.plot_metrics_comparison(train_metrics, test_metrics)
    
    # Gráfico de progresso do treinamento
    plot.plot_training_progress(model)

    # Gráficos de previsões vs valores reais
    plot.plot_predictions_vs_actual(dates_train, y_train_actual, y_train_pred_rescaled, 
                              f"Conjunto de Treino - {TICKER}")

    plot.plot_predictions_vs_actual(dates_test, y_test_actual, y_test_pred_rescaled, 
                              f"Conjunto de Teste - {TICKER}")
    
    print("✅ Análise completa!")
    
    # Salvar modelo
    torch.save(model.state_dict(), f'lstm_model_{TICKER}.pth')
    print(f"💾 Modelo salvo como lstm_model_{TICKER}.pth")

if __name__ == "__main__":
    main()