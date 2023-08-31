from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import model

app = Flask(__name__, template_folder="/home/inova/Documentos/GPTMedico/templates")
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/perguntar", methods=["POST"])
def perguntar():
    try:  
        pergunta = request.json['pergunta']

        resposta = model.responder(pergunta)

        return {"resposta": resposta}
  
    except Exception as e:
        print(f"Erro ao responder: {e}")  
        return {"erro": "Erro na resposta"}, 500

if __name__ == "__main__":
    app.run(debug=True)
