# Tech Challenge 04 - API de Previsão de Séries Temporais

Esta API utiliza uma Rede Neural LSTM para previsão de séries temporais, permitindo treinar modelos e realizar previsões com base em dados históricos.

## Configuração

Antes de executar a API, crie um arquivo `.env` na raiz do projeto com os seguintes campos:

```ini
PATH_MODEL_JSON="models.json"
PATH_MODEL="modelos_treinados"

aws_access_key_id=""
aws_secret_access_key=""
aws_session_token=""
```

Preencha as credenciais AWS caso necessite de acesso a serviços da AWS.

## Instalação

  1. Clone o repositório:

  ```sh
  git clone <URL_DO_REPOSITORIO>
  cd <NOME_DO_REPOSITORIO>
  ```
  
  2. Crie um ambiente virtual:

  ```sh
  python -m venv venv
  venv\Scripts\activate
  ```

  3. Instale as dependências:

  ```sh
  pip install -r requirements.txt
  ```

  ## Uso

  ## Iniciar a API

  Execute o seguinte comando para iniciar a API com FastAPI:

  ```sh
  uvicorn main:app --reload
  ```

  A documentação Swagger estará disponível em http://127.0.0.1:8000/docs.
  
  ## Endpoints

    GET /models

  Retorna os modelos disponíveis.

    POST /models/train

  Treina um novo modelo com os seguintes parâmetros:

  ```json
  {
    "ticker": "AAPL",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
    "train_size": 0.8,
    "sequence_length": 30,
    "num_epochs": 100
  }
  ```

  Retorna as métricas do modelo treinado.

    POST /models/predict

  Realiza previsões usando um modelo treinado.

  ```json
  {
    "model_id": "123",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "days": 30,
    "sequence_length": 30
  }
  ```

  Retorna os valores previstos.
  
  ## Estrutura do Código

  * `model/` - Contém a implementação do modelo LSTM.
  * `repository/` - Lida com armazenamento e recuperação dos modelos treinados.
  * `util/` - Contém funções auxiliares para manipulação de dados.
  * `main.py` - Ponto de entrada da aplicação.
