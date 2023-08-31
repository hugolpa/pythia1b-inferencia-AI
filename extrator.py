import os
import xml.etree.ElementTree as ET  
import csv

dataset_dir = '/home/inova/Documentos/GPTMedico/MedQuAD/1_CancerGov_QA'

# Renomeia os arquivos XML
xml_files = os.listdir(dataset_dir)
xml_files.sort()

for index, file in enumerate(xml_files):
    os.rename(os.path.join(dataset_dir, file), 
              os.path.join(dataset_dir, f'{index}.xml'))

# Abre o arquivo CSV para escrita   
with open('medquad_data.csv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow(['Question', 'Answer']) 

    # Loop pelos arquivos XML renomeados
    for i in range(len(xml_files)):
        
        # Lê o XML
        tree = ET.parse(os.path.join(dataset_dir, f'{i}.xml')) 
        root = tree.getroot()

        # Extrai perguntas e respostas
        qapairs = root.find('QAPairs')
        
        for qapair in qapairs:
            question = qapair.find('Question').text
            answer = qapair.find('Answer').text
            
            # Escreve no CSV
            writer.writerow([question, answer])

print('Arquivo CSV criado com sucesso!')