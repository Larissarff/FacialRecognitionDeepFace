from deepface import DeepFace
import matplotlib.pyplot as plt

path = r"./DataFrame/tests/teste_02.jpg"

obj = DeepFace.analyze(
    img_path=path,
    actions=["age", "gender", "race", "emotion"],
    enforce_detection=False,
)

print(obj)

for i in obj:
    print(f"Age: {i['age']}")
    print(f"Gender: {i['gender']}")
    print(f"Race: {i['race']}")
    print(f"Emotion: {i['emotion']}")