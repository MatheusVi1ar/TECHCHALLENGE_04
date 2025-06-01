import io
#import os
import boto3

#aws_access_key_id="ASIAQ26XKOCGWK4R5QTA"
#aws_secret_access_key="Jjus2eNrzGqX0S6e71Pver9GsVPCJ3uDHsE/3kMe"
#aws_session_token="IQoJb3JpZ2luX2VjEPv//////////wEaCXVzLXdlc3QtMiJHMEUCIFHLFxOZtxhviW2pQVUhor4MZYRv3UfE4amhBmnLw/9pAiEAsR3O9nffllFqR6cbzOG2ONjrIORjMpOJQ/gvseVj6UQqvwIIxP//////////ARABGgwwNTc4OTY3NTk0MzciDErhSUDdFDXyB45rACqTAuXuFarj91xjUKw/117WQ20wzgN2rouoBWY2QUvnkm2Vvum5m8rKJVdH05YozayaPuli419Nw1KBk/ig97esxMesfpCsJLT+LWAbqcJExTfRQxkDbyl8d9hu8QWbmqIfaJYiDjSoqX6BjafjwiWOojEgDmful4TReflbtsRiMbAcX4tUk3qOP1n59m3QfNu+xtMt9R7F0wx0Bik/CNyGXC/8f21xJRrTaOEZ6krafH9ZRSh2X9G5MuONVdKj9AcVnOXgYhfH2h2CGBprG8HnOyrenTAvora4CzPkTFWLr2jKiQBmxMsIv2qKrOngiS2VDyYJ6i0Lj0iwkGob81ZPnxtflvTCBkHSf+yzU9PuESmQKZ3yMNyb7cEGOp0BNAlu21B3AfsdtatB/4wmgA3zY5DN0KYgE56goYUlvy1ANovg/FZOvPlfRX+kCQTnGSrsKdlQL8rPpXG9j2eySXegneT9VpFUSbcN7xYR6Fjvq+cNf+WWbVleRKTbbAH/YXP/3tDTHnaeU7eM3kltPInpEWS/wwcN6PC2cktZAhzN8VUpf369uoaLVfVWaCNNNHT35vRe3U1jB3Os0w=="

def gerar_s3_client():
    # Upload Parquet file to S3
    #DesComentar para publicar
    #session = boto3.Session(
    #    aws_access_key_id=os.getenv("aws_access_key_id"),
    #    aws_secret_access_key=os.getenv("aws_secret_access_key"),
    #    aws_session_token=os.getenv("aws_session_token"),
    #)

    #DesComentar para publicar
    #s3_client = session.client('s3')
    #comentar para publicar
    s3_client = boto3.client('s3') 

    bucket_name = 'tc04-bucket-s3'

    return s3_client, bucket_name

def gravar_s3(file_data, s3_file_name):
    """Envia um arquivo JSON para o bucket S3 especificado."""     
    s3_client, bucket_name = gerar_s3_client()

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
    s3_client, bucket_name = gerar_s3_client()

    # Carregar JSON
    buffer_json = io.BytesIO()
    s3_client.download_fileobj(bucket_name, s3_json_path, buffer_json)
    buffer_json.seek(0)
    
    print(f"JSON carregado de S3://{bucket_name}/{s3_json_path}")
    
    return buffer_json
    
    # Função adicional para carregar o modelo do S3
def carregar_lstm_s3(s3_model_path, s3_scaler_path):
    """Carrega o modelo e scaler do S3"""
    s3_client, bucket_name = gerar_s3_client()

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