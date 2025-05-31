from pydantic import BaseModel, Field

class Model(BaseModel):
    id: int = Field(..., description="Identificador único para o modelo")
    ticker: str = Field(..., description="Símbolo da ação")
    start_date: str = Field(..., description="Data de início para os dados no formato YYYY-MM-DD")
    end_date: str = Field(..., description="Data de término para os dados no formato YYYY-MM-DD")
    file: str = Field(..., description="Caminho do arquivo para o modelo")
    scaler_file: str = Field(..., description="Caminho do arquivo para o scaler de normalização")
    
class TrainRequest(BaseModel):
    ticker: str = Field(..., description="Símbolo da ação a ser treinada")
    start_date: str = Field(..., description="Data de início para os dados no formato YYYY-MM-DD")
    end_date: str = Field(..., description="Data de término para os dados no formato YYYY-MM-DD")
    train_size: float = Field(0.8, description="Proporção dos dados usados para treinamento (0 a 1)")
    sequence_length: int = Field(30, description="Número de dias em cada sequência para o modelo LSTM")
    
class PredictionRequest(BaseModel):
    model_id: int = Field(..., description="ID do modelo a ser usado para previsão")
    start_date: str = Field(..., description="Data de início para os dados no formato YYYY-MM-DD")
    end_date: str = Field(..., description="Data de término para os dados no formato YYYY-MM-DD")
    days: int = Field(1, description="Número de dias para previsão")
    sequence_length: int = Field(30, description="Número de dias em cada sequência para o modelo LSTM")