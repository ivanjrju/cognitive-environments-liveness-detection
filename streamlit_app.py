import streamlit as st
import cv2
import boto3
import numpy as np

"""
# Cognitive Environments
## Liveness Detection

O detector de Liveness (Vivacidade) tem por objetivo estabelecer um índice que atesta o quão 
confiável é a imagem obtida pela câmera.
Imagens estáticas, provindas de fotos manipuladas, são os principais focos de fraude neste tipo de validação.
Um modelo de classificação deve ser capaz de ler uma imagem da webcam, classificá-la como (live ou não) e 
exibir sua probabilidade da classe de predição.

"""

client = boto3.client('rekognition', region_name='us-east-1')

project_arn='arn:aws:rekognition:us-east-1:465575464224:project/detect_face_liveness/1722564559345'
version_name='detect_face_liveness.2024-08-01T23.29.48'
model_arn='arn:aws:rekognition:us-east-1:465575464224:project/detect_face_liveness/version/detect_face_liveness.2024-08-01T23.29.48/1722565788918'

project_version_running_waiter = client.get_waiter('project_version_running')
project_version_running_waiter.wait(ProjectArn=project_arn, VersionNames=[version_name])


uploaded_file = st.file_uploader('Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_analyze = uploaded_file.getvalue()

camera = st.camera_input("Tire sua foto", help="Lembre-se de permitir ao seu navegador o acesso a sua câmera.")
if camera is not None:
    img_analyze = camera.getvalue()
    imagem = cv2.imdecode(np.frombuffer(img_analyze, np.uint8), cv2.IMREAD_COLOR)

if camera or uploaded_file:
    with st.spinner('Classificando imagem...'):

        response = client.detect_custom_labels(
                Image={'Bytes': img_analyze},
                    MinConfidence=50,
                    ProjectVersionArn=model_arn)
            
        label = response['CustomLabels'][0]['Name']
        confidence = response['CustomLabels'][0]['Confidence']

        if(label == "real"):
            st.success("Probabilidade da foto ser real é de {:.2f}%".format(confidence))
        else:
            st.error("Probabilidade da foto ser falsa é de {:.2f}%".format(confidence))
