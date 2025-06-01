# Usar uma imagem oficial do Python
FROM python:3.12-slim

# Definir o diretório de trabalho no contêiner
WORKDIR /app

# Copiar o arquivo de requisitos e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar os arquivos da aplicação para o contêiner
COPY . .

# Expor a porta usada pela API
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
