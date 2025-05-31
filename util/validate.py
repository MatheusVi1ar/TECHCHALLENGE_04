from datetime import datetime, timedelta
from util import util


def validate_train_request(train_request):
    """Valida os parâmetros do pedido de treinamento"""
    if not train_request.ticker:
        raise Exception("O ticker não pode ser vazio.")
    if not train_request.start_date or not train_request.end_date:
        raise Exception("As datas de início e término não podem ser vazias.")
    if train_request.train_size <= 0 or train_request.train_size >= 1:
        raise Exception("O tamanho do treinamento deve ser um valor entre 0 e 1.")
    try:
        start_date_obj = datetime.strptime(train_request.start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(train_request.end_date, '%Y-%m-%d')
        if start_date_obj >= end_date_obj:
            raise ValueError("A data de início deve ser anterior à data de término.")
        if end_date_obj - start_date_obj < timedelta(days=train_request.sequence_length):
            raise ValueError("O intervalo entre as datas de início e término deve ser maior ou igual ao comprimento da sequência.")
    except Exception as e:
        raise Exception(str(e))
    
def validate_prediction_request(predict_request):
    """Valida os parâmetros do pedido de previsão"""
    if not predict_request.model_id:
        raise Exception("O ID do modelo não pode ser vazio.")
    if util.consultar_modelo(predict_request.model_id) is None:
        raise Exception(f"Modelo com ID {predict_request.model_id} não encontrado.")
    if not predict_request.start_date or not predict_request.end_date:
        raise Exception("As datas de início e término não podem ser vazias.")
    if predict_request.days <= 0:
        raise Exception("O número de dias para previsão deve ser um valor positivo.")
    try:
        start_date_obj = datetime.strptime(predict_request.start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(predict_request.end_date, '%Y-%m-%d')
        if start_date_obj >= end_date_obj:
            raise ValueError("A data de início deve ser anterior à data de término.")
        if end_date_obj - start_date_obj < timedelta(days=predict_request.sequence_length):
            raise ValueError("O intervalo entre as datas de início e término deve ser maior ou igual ao comprimento da sequência.")
    except Exception as e:
        raise Exception(str(e))