import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator

# Carrega modelo e tokenizer
modelo = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-deduped")
tokenizador = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
tradutor = Translator() 

# Função para gerar resposta
def responder(pergunta):

  try:
    # Traduz pergunta para inglês
    pergunta_en = tradutor.translate(pergunta, dest="en").text

    # Pré-processa 
    pergunta_en = pergunta_en.strip().lower()

    # Gera resposta em inglês 
    entrada_ids = tokenizador.encode(pergunta_en, return_tensors="pt")
    saida_ids = modelo.generate(entrada_ids, max_length=40)

    # Traduz resposta para português
    resposta_en = tokenizador.decode(saida_ids[0], skip_special_tokens=True)
    resposta_pt = tradutor.translate(resposta_en, dest="pt").text

    return resposta_pt
  
  except Exception as e:
    # Log de erros
    logging.error(f"Erro ao gerar resposta: {str(e)}")
    return "Desculpe, não entendi a pergunta. Poderia reformular?"
    
3