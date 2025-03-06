"""# Setup"""
import ollama
import pandas as pd
import numpy as np
import matplotlib as plt
import os, errno
import pickle
import re
from random import shuffle, randint



"""# ANOTANDO COM LLM"""

# o keyphrase tanto faz, pois a parte dos documnetos é a mesma perante todos os kyphrases
# assim, como usamos apenas o y (deriva de doc) podemos nos basear em qualquer keyphrase
system_prompt_dict = {
"CSTR":'''You are a text classification AI. Your job is to classify Documents from the CSTR (Computer Science Technical Reports) collection, composed by abstracts and technical reports published in the Department of Computer Science at University of Rochester from 1991 to 2007. The documents belong to 4 areas: Natural Language Processing, Robotics/Vision, Systems, and Theory.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.''',

"Dmoz_Computers":
'''You are a text classification AI. Your job is to classify Documents from the Dmoz-Computers-500 collection that, composed by web pages of the computers category extracted from DMOZ - Open Directory Project. The document class are the subcateries of the computers category.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.''',

"Dmoz_Health":'''You are a text classification AI. Your job is to classify Documents from the Dmoz-Health-500 collection, composed by web pages of the health category extracted from DMOZ - Open Directory Project. The document class are the subcateries of the health category.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.
Below there are examples to your task:
{1}''',

"Dmoz_Science":'''You are a text classification AI. Your job is to classify Documents from the Dmoz-Science-500 collection, composed by web pages of the science category extracted from DMOZ - Open Directory Project. The document class are the subcateries of the science category.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.
Below there are examples to your task:
{1}''',

"Dmoz_Sports":'''You are a text classification AI. Your job is to classify Documents from the Dmoz-Sports-500 collection, composed by web pages of the sports category extracted from DMOZ - Open Directory Project. The document class are the subcateries of the sports category.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.
Below there are examples to your task:
{1}''',

"Industry_Sector":'''You are an text classification AI. Your job is to classify Documents from the Industry-Sector collection, composed by web pages of companies from various economic sectors.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.''',

"NSF":'''You are a text classification AI. Your job is to classify Documents from the NSF (National Science Foundation) collection, composed by abstracts of grants awarded by the National Science Foundation8 between 1999 and August 2003.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.
Below there are examples to your task:
{1}''',

"SyskillWebert":'''You are a text classification AI. Your job is to classify Documents from the SyskillWebert collection, composed by web pages about bands, sheeps, goats,and biomedicals.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.''',

"classic4":'''You are a text classification AI.Your job is to classify Documents from the Classic4 collection, composed by 4 distinct collections: CACM (titles and abstracts from the journal Communications of the ACM), CISI (information retrieval papers), CRANFIELD (aeronautical system papers), and MEDLINE (medical journals).
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.
Below there are examples to your task:
{1}''',

"re8":'''You are a text classification AI. Your job is to classify Documents from the Re8 collection, composed by articles from Reuters-21578 collection.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.
Below there are examples to your task:
{1}''',

"review_polarity":'''You are a text classification AI. Your job is to classify Documents from the Review-Polarity collection, composed by 1000 positive reviews and 1000 negative reviews about movies.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.''',

"webkb_parsed":'''You are a text classification AI. Your job is to classify Documents from the WebKB colleciton, composed by web pages collected from computer science departments of various universities in January 1997 by the World Wide Knowledge Base (WebKb) project of the CMU Text Learning Group.
The collection classes are the following:
{0}

You must not talk to the user. Limit your response to only the predicted class.'''
}
keyphrase= "keyphrase2"
sample_sizes = [100]
iterations = range(10)
# datasets = ["Dmoz_Computers","Dmoz_Health","Dmoz_Science","Dmoz_Sports","Industry_Sector","NSF","classic4","re8","review_polarity","SyskillWebert","webkb_parsed"]#  "CSTR"
# n_rotulated = [1,5,10,20,30]
datasets = ["CSTR"]
n_rotulated = [1,5,10,20]

output_datasets = "output_datasets_100"
input_samples = ""
# output_datasets = "output_datasets_few_shot"
# input_samples = "output_datasets_zero_shot"
#output_datasets = "output_datasets_zero_shot"
#input_samples = ""

def findFirstOccurence(wordList, string, startIndex=0):
    x = re.search('|'.join(wordList), string[startIndex:])
    if x:
        return x.group()
    else:
        return ''

for dataset_name in datasets:
    # Load dataset
    df = pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/processed_datasets/"+dataset_name+"/"+dataset_name+".csv")
    df = df[['id','text','class']]

    print(df.head())

    # Creating labels for classification
    # the dictionary is a numeric representation for each possible "document" class
    lista_classes = df["class"].unique()

    labels_dit = {}
    for i,classe in enumerate(lista_classes):
        labels_dit[classe] = i
    print(labels_dit,'\n')

    for sample_size in sample_sizes:
        for rotulated in n_rotulated:
            with open('/home/vitor_vasconcelos/workdir/mestrado_v3/processed_datasets/'+dataset_name+'/'+keyphrase+'/graph.pkl','rb') as f:
                aux =  pickle.load(f)
                y = aux[3]
            
            # cria diretorio para maks e fake y alterados
            try:
                os.makedirs("/home/vitor_vasconcelos/workdir/mestrado_v3/"+output_datasets+"/"+dataset_name+'/new_masks_and_y_'+str(sample_size))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            # cria diretorio para acertos e erros
            try:
                os.makedirs('/home/vitor_vasconcelos/workdir/mestrado_v3/'+output_datasets+'/'+dataset_name+'/acertos_'+str(sample_size))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            for iteration in iterations:
                with open('/home/vitor_vasconcelos/workdir/mestrado_v3/processed_datasets/'+dataset_name+'/masks/mask_rot'+str(rotulated)+'_'+str(iteration)+'.pkl', 'rb') as f:
                    aux = pickle.load(f)
                    train_mask, val_mask, test_mask = aux[0],aux[1],aux[2]

                # argupa lista de ids da mascara de treino
                lista_train_id =[]
                for i,mask in enumerate(train_mask):
                    if mask == 1:
                        lista_train_id.append(i)

                # Filtra um dataframe refrente a operaçõe da mascara de teste
                df_test = df.drop(lista_train_id)

                # Filtra um dataframe refrente a operaçõe da mascara de treino
                df_train = df[df['id'].isin(lista_train_id)]

                # Seleciona samples aleatorias igualmente para cada classe do dataframe (caso a o numero de documentos daclasse seja inferior ao samplesize são selecionada metade para llm)
                if input_samples == "":
                    df_sample = df_sample = df_test.groupby('class', as_index=False)[['id','text','class']]\
                            .apply(lambda x: x.sample(sample_size) if (len(x)>=sample_size) else x.sample(len(x)//2)).reset_index()
                else:
                    with open('/home/vitor_vasconcelos/workdir/mestrado_v3/'+input_samples+'/'+dataset_name+'/acertos_'+str(sample_size)+'/acertos'+str(rotulated)+'_'+str(iteration)+'.pkl','rb') as f:
                        aux =  pickle.load(f)
                        df_sample = aux[0]


                # Inico da classifição do modelo LLM (Petals)
                acertos = 0
                dit_acertos = {}
                for i,classe in enumerate(lista_classes):
                    dit_acertos[classe] = 0
                
                # cria o prompt do systema com exemplos de classificacoes do dataset de treino
                lista_str = ""
                # Cria o dataframe de expleos para o prompt
                df_examples = df_train\
                                .groupby('class', as_index=False)[['id','text','class']]\
                                .apply(lambda x: x.sample(1))

                if len(df_examples)>=5:
                    chosen_idx = np.random.choice(len(df_examples), replace=False, size=5)
                    df_examples = df_examples.iloc[chosen_idx]
                
                for train_index,train_sample in df_examples.iterrows():
                    lista_str = lista_str + f"Document: {train_sample['text']}\nClass:{train_sample['class']}\n\n"

                str_lista_classes = '['+ ', '.join(lista_classes) +']'
                system_prompt = system_prompt_dict[dataset_name].format(str_lista_classes, lista_str)
                print(system_prompt)

                # inicia aplicação

                idx_list = df_sample.index.to_list()  # get index to a list
                shuffle(idx_list)              # shuffle the list using `random.shuffle()`

                for index in idx_list:

                    prompt ='''
Classify the Document into one of the following classes:
{0}
Limit your response to only the predicted class.
Document: {1}
Class:
'''.format(str_lista_classes, df_sample.at[index,'text'])

                # print('system_prompt:\n',system_prompt)
                # print('prompt:\n',prompt)
                # print('class:\n',df_sample.at[index,'class'])

                    response = ollama.generate(model = "llama3.1",
                                    system = system_prompt,
                                    prompt = prompt,
                                    options={"num_predict": 25,
                                            "temperature":0.3})
                    output = response['response']
                    # response = ollama.generate(model="llama3", prompt= prompt )
                    # print(type(response))
                    # print(response['response'])

                    output_pred = findFirstOccurence(lista_classes,output)

                    # Salva predição no dataframe de samples e contabiliza erros e acertos
                    comemora = "ERROU ;-;"
                    if output_pred == df_sample.at[index,'class']:
                        # Salva o output e a classe correta
                        df_sample.at[index, 'pred_class'] = output_pred
                        df_sample.at[index, 'pred'] = output
                        # contabiliza acerto
                        acertos+=1
                        dit_acertos[df_sample.at[index,'class']]+=1
                        comemora = "ACERTOU :)"
                    # Caso a predicao nao exista, damos uma segunda chance para LLM 
                    elif output_pred == '':
                        #print(' - WARNING - it{0}-{1}:  CLASSE PREDITA NAO EXISTE. Tentando novamente...'.format(iteration, index))
                        response = ollama.generate(model = "llama3.1",
                                    system = system_prompt,
                                    prompt = prompt,
                                    options={"num_predict": 25,
                                            "temperature":0.3})
                        output = response['response']

                        output_pred = findFirstOccurence(lista_classes,output)
                        
                        if output_pred == '':
                            #print(' - WARNING - it{0}-{1}:  CLASSE PREDITA NAO EXISTE. Sem predicao.'.format(iteration, index))
                            df_sample.at[index, 'pred_class'] = None
                            df_sample.at[index, 'pred'] = output
                        elif output_pred == df_sample.at[index,'class']:
                            # Salva o output e a classe correta
                            df_sample.at[index, 'pred_class'] = output_pred
                            df_sample.at[index, 'pred'] = output
                            # contabiliza acerto
                            acertos+=1
                            dit_acertos[df_sample.at[index,'class']]+=1
                            comemora = "ACERTOU :)"
                        else:
                            # Salva o output e a classe errada
                            df_sample.at[index, 'pred_class'] = output_pred
                            df_sample.at[index, 'pred'] = output
                    # Caso tenha predito a classe errada
                    else:
                        # Salva o output e a classe errada
                        df_sample.at[index, 'pred_class'] = output_pred
                        df_sample.at[index, 'pred'] = output
                    
                    #print('it{0}-{1}: predito:{2}  |  Classe real:{3}  |  Acertos={4}  |  {5}'.format(iteration, index, output,df_sample.at[index,'class'], acertos,comemora))
                    # print(dit_acertos)
                    # df_sample.at[index, 'pred_class'] = output.strip('@Class:')
                    # df_sample.at[index, 'pred_class'] = output

                print(f'{dataset_name}_{rotulated}, size_{sample_size}, {iteration}° ITERATION RESULTS:')
                pocentagem_acertos = acertos/len(df_sample)
                print(pocentagem_acertos)
                print(dit_acertos)

                # Ajustando
                for index,sample in df_sample.iterrows():
                    #ajusta mascaras
                    train_mask[int(df_sample['id'][index])]=True
                    test_mask[int(df_sample['id'][index])]=False

                    # ajusta Y
                    if df_sample['pred_class'][index] in lista_classes:
                        y[int(df_sample['id'][index])] = labels_dit[df_sample['pred_class'][index]]
                    #Caso clase predita nao exista tente novamente
                    else:
                        # Seta o y como uma classe nao existente para possivel tratamento posteriormente
                        # Salva o output e a classe localizada
                        y[int(df_sample['id'][index])] = len(lista_classes)
                        # y[int(df_sample['id'][index])] = randint(0, (len(lista_classes()-1)))

                # print('train_mask:',train_mask)
                # print('test_mask:',test_mask)
                # print('y:',y)
                print('df_sample:\n',df_sample)

                # Salva masks e Y alterados
                with open('/home/vitor_vasconcelos/workdir/mestrado_v3/'+output_datasets+'/'+dataset_name+'/new_masks_and_y_'+str(sample_size)+'/mask_rot'+str(rotulated)+'_'+str(iteration)+'.pkl', 'wb+') as f:
                    aux = [train_mask,val_mask,test_mask, y]
                    pickle.dump(aux, f)

                # Salva acertos
                with open('/home/vitor_vasconcelos/workdir/mestrado_v3/'+output_datasets+'/'+dataset_name+'/acertos_'+str(sample_size)+'/acertos'+str(rotulated)+'_'+str(iteration)+'.pkl', 'wb+') as f:
                    aux = [df_sample,dit_acertos,acertos,pocentagem_acertos]
                    pickle.dump(aux, f)

