from transformers import AutoModelForCausalLM, AutoTokenizer

# Carrega modelo finetuned
modelo = AutoModelForCausalLM.from_pretrained("./medquad_finetuned")
tokenizador = AutoTokenizer.from_pretrained("./medquad_finetuned")

# Usa para inferência
entrada_ids = tokenizador("Quais causam dor de cabeça?", return_tensors="pt").input_ids
saidas_ids = modelo.generate(entrada_ids)
resposta = tokenizador.decode(saidas_ids[0])

print(resposta)