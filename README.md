# EleutherAI-pythia-1b-deduped-inference
## Inferencia em modelo "EleutherAI/pythia-1b-deduped", tradução portgues-ingles, user interface. 
## Link no hugging face: [EleutherAI/pythia-1b-deduped](https://huggingface.co/EleutherAI/pythia-1b-deduped)

## Estrutura do Projeto

O projeto está organizado da seguinte maneira:

- **static**: Diretório para arquivos estáticos JavaScript. Funcionalidade "spinner de carregamento" para indicar visualmente que a ação resposta está em andamento.
- **templates**: Diretório para arquivos de modelos HTML usados pelo Flask para renderizar página de pergunta. (css feito com bootstrap)
- **app.py**: O código-fonte principal da aplicação Flask, servidor.
- **model.py**:  contém implementações do modelo, funcionalidades como tradução, max_lenght (tamanho da resposta).
- **requirements-comments.txt**:  Dependências do projeto.

## Requisitos de Instalação

Para executar este projeto localmente, você precisará das seguintes dependências Python:

- Flask
- Transformers
- torch
- Googletrans
## Deixei pois vai ser necessário para o finetune
- python-dotenv
- requests
- nltk

Recomendo criar um ambiente virtual para isolar as dependências do projeto. Você pode usar a ferramenta `virtualenv` para isso.

```bash
# Crie um ambiente virtual (certifique-se de estar na raiz do projeto)
# Ative o ambiente virtual (dependendo do seu sistema operacional)
python -m venv myenv  # Crie o ambiente virtual com o nome "myenv"
source myenv/bin/activate  # Ative o ambiente virtual (Linux/macOS)
myenv\Scripts\activate  # Ative o ambiente virtual (Windows)


# Instale as dependências a partir do requirements.txt
pip install -r requirements-comments.txt
# Executando o Projeto
Para iniciar a aplicação, você pode usar o seguinte comando:

python app.py
````
A aplicação estará disponível em http://localhost:5000

# Imagem docker se cria com o comando
```bash
sudo docker build -t docker-image-inference .
sudo docker-compose up
# Comando para usar executar uma inferência(fazer uma pergunta e receber uma resposta)
curl -X POST -F "question=o que é diabetes" http://localhost:5050/ask
