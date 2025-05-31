import io
import os
import boto3

def gravar_s3(file_data, s3_file_name):
    """Envia um arquivo JSON para o bucket S3 especificado.""" 
    # Upload Parquet file to S3 
    # #Comentar para publicar
    session = boto3.Session(
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        aws_session_token=os.getenv("aws_session_token"),
   )

    s3_client = session.client('s3')#Comentar para publicar
#    s3_client = boto3.client('s3') #Descomentar para publicar

    bucket_name = 'tc04-bucket-s3'

    # Verificar o tipo de dados e processar adequadamente
    if isinstance(file_data, io.BytesIO):
        # Para buffers BytesIO (modelos e scalers)
        s3_client.upload_fileobj(file_data, bucket_name, s3_file_name)
    elif isinstance(file_data, str):
        # Para strings JSON
        buffer = io.BytesIO(file_data.encode('utf-8'))
        s3_client.upload_fileobj(buffer, bucket_name, s3_file_name)
    elif isinstance(file_data, bytes):
        # Para dados em bytes
        buffer = io.BytesIO(file_data)
        s3_client.upload_fileobj(buffer, bucket_name, s3_file_name)
    else:
        raise TypeError(f"Tipo de dados não suportado: {type(file_data)}")   

    print(f"File successfully uploaded to S3://{bucket_name}/{s3_file_name}")
   
# Função para carregar JSON do S3
def carregar_json_s3(s3_json_path):
    """Carrega um arquivo JSON do S3"""
    # Comentar para publicar
    session = boto3.Session(
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        aws_session_token=os.getenv("aws_session_token"),
    )
    s3_client = session.client('s3')  # Comentar para publicar
    # s3_client = boto3.client('s3')  # Descomentar para publicar
    
    bucket_name = 'tc04-bucket-s3'
    
    # Carregar JSON
    buffer_json = io.BytesIO()
    s3_client.download_fileobj(bucket_name, s3_json_path, buffer_json)
    buffer_json.seek(0)
    
    print(f"JSON carregado de S3://{bucket_name}/{s3_json_path}")
    
    return buffer_json
    
    # Função adicional para carregar o modelo do S3
def carregar_lstm_s3(s3_model_path, s3_scaler_path):
    """Carrega o modelo e scaler do S3"""
    # Comentar para publicar
    session = boto3.Session(
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        aws_session_token=os.getenv("aws_session_token"),
    )
    s3_client = session.client('s3')  # Comentar para publicar
    # s3_client = boto3.client('s3')  # Descomentar para publicar
    
    bucket_name = 'tc04-bucket-s3'
    
    # Carregar modelo
    buffer_model = io.BytesIO()
    s3_client.download_fileobj(bucket_name, s3_model_path, buffer_model)
    buffer_model.seek(0)
    
    # Carregar scaler
    buffer_scaler = io.BytesIO()
    s3_client.download_fileobj(bucket_name, s3_scaler_path, buffer_scaler)
    buffer_scaler.seek(0)
    
    print(f"Modelo carregado de S3://{bucket_name}/{s3_model_path}")
    print(f"Scaler carregado de S3://{bucket_name}/{s3_scaler_path}")
    
    return buffer_model, buffer_scaler