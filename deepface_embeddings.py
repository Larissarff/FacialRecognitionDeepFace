from matplotlib import pyplot as plt
from deepface import DeepFace

models = [
    "VGG-Face",
    "Facenet",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet",
]
backends = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "fastmtcnn",
    "retinaface",
    "mediapipe",
    "yolov8",
    "yunet",
    "centerface",
]

img_path1 = 'DataFrame/tests/teste_02.jpg'
img_path2 = 'DataFrame/tests/teste_03.jpg'

# Gerar usando a função represent do DeepFace para ambas imagens
embedding1 = DeepFace.represent(img_path=img_path1, model_name=models[0], 
						 detector_backend=backends[7])[0]['embedding']
embedding2 = DeepFace.represent(img_path=img_path2, model_name=models[0], 
						 detector_backend=backends[7])[0]['embedding']
# Plotando graficos separadamente
plt.figure(figsize=(12, 6))

# Grafico 1 - Scatter plot para exibir apenas os pontos
plt.subplot(2, 1, 1)
plt.scatter(range(len(embedding1)), embedding1, marker='.', color='coral')
plt.xlabel('Índice do Embedding')
plt.ylabel('Valor do Embedding')
plt.title('Grafico de Dispersão do Embedding - Imagem 1')

# Grafico 2 - Scatter plot para exibir apenas os pontos
plt.subplot(2, 1, 2)
plt.scatter(range(len(embedding2)), embedding2, marker='*', color='lightseagreen')
plt.xlabel('Índice do Embedding')
plt.ylabel('Valor do Embedding')
plt.title('Grafico de Dispersão do Embedding - Imagem 2')

#Grafico da imagem 1 e 2 sobrepostos
plt.figure(figsize=(12, 6))
plt.scatter(range(len(embedding1)), embedding1, marker='.', color='coral',
            label='Imagem 1 - Embedding')
plt.scatter(range(len(embedding2)), embedding2, marker='*', color='lightseagreen',
            label='Imagem 2 - Embedding')
plt.xlabel('Índice do Embedding')
plt.ylabel('Valor do Embedding')
plt.title('Grafico de Dispersão do Embedding - Imagens 1 e 2 Sobrepostos')

plt.tight_layout()
plt.show()