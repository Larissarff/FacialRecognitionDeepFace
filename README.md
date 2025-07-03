## inicializando o projeto:
### Crie seu projeto python, criando seu venv e ative ele.

ativando o ambiente .venv:
    
    python3 -m venv venv // criação
    source .venv/bin/activate // ativação

    
    baixando as bibliotecas (linux):
    
    sudo apt update
    sudo apt install cmake
    sudo apt install build-essential python3-dev

    pip uninstall tensorflow
    pip install tensorflow==2.10.1
    pip show tensorflow
        
    ''' deve retornar: Version: 2.10.1 '''
  

    libs:
    
    pip install deepface
   
    pip install ultralytics

    pip install dlib --no-cache-dir

    pip install facenet-pytorch
 
    pip install mediapip
    
    rodar o projeto:
    python main.py 