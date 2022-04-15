#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/dilsousa/DataScience/blob/main/Dil_DesafioInferencia.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 

# ##Importar as bibliotecas que precisamos

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# ##Nesse caso, armazenei os *dataset's* no meu drive, por isso chamei eles assim

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ##Aqui apenas criei um objeto no python para armazenar os *dataset's*.

# 

# In[ ]:


demo = pd.read_csv('/content/drive/My Drive/Tera/Datasets/DEMO_PHQ.csv')
pag_hei = pd.read_csv('/content/drive/My Drive/Tera/Datasets/PAG_HEI.csv')


# ## Verificando os *dataset's*...

# In[ ]:


demo.info()


# In[ ]:


pag_hei.info()


# ##Nessa análise foi definido que o banco "demo" será o principal e o "pag_hei" deve se juntar a ele.
# ##Para tanto, o comando abaixo mescla os dois bancos pela esquerda. O "demo" está na esquerda então o "pag_hei" deve se juntar a ele.

# In[ ]:


db = demo.merge(pag_hei, on = 'SEQN', how = 'left')
db.info()


# In[ ]:


db.head()


# ##Nesse comando observa-se a quantidade de faltantes do banco, alguns sairam com 9,34% outros 5,14%. Deve-se avaliar se os dados faltantes são prejudiciais para a análise. Nesse momento foi decidido que os dados faltantes não representam problemas para os resultados.

# In[ ]:


db.isnull().mean().round(4)*100


# ## A mesma análise agora em números e não porcentagem

# In[ ]:


db.isna().sum()


# ## Nesse passo o comando 'DROP' excluiu apenas nessa linha a coluna SEQN e descreve o comportamento das demais...
# ##Ex: Qual é o .99 percentil da coluna DPQ010? A resposta é 3, significando que 99% das pessoas têm 3 ou menos.
# 

# In[ ]:


db.drop(columns = ['SEQN']).describe(percentiles = [.25, .5, .75, .95, .99]).round(2)


# In[ ]:


db[['DPQ010',
    'DPQ020',
    'DPQ030',
    'DPQ040',
    'DPQ050',
    'DPQ060',
    'DPQ070',
    'DPQ080',
    'DPQ090']].agg(['value_counts'])


# ## Continua-se observando o *dataset* em busca de informações que possam ajudar na análise e construção de hipóteses

# ##O comando abaixo mostra as diferentes etnias dos participantes.
# O parâmetro sort = False é para que o value_counts não ordene por numeração. Ex: 1,2,3,4...

# In[ ]:


db['RIDRETH1'].value_counts(sort = False)

# RIDRETH1 com códigos errados no desafio
# 1	Mexican American	
# 2	Other Hispanic
# 3	Non-Hispanic White	
# 4	Non-Hispanic Black	
# 5	Other Race - Including Multi-Racial


# ##Nível de escolaridade

# In[ ]:


db[["DMDEDUC"]].value_counts(sort = False)


# ## Renda por ano dos participantes

# In[ ]:


db[["INDFMINC"]].value_counts(sort = False)


# ##Aderência aos exercícios físicos

# In[ ]:


db[["ADHERENCE"]].value_counts(sort = False)


# ## No *dataset* foram encontradas categorias na coluna 'RIDRETH1' que podem ser mescladas. Categoria 2 (Other Hispanic) e 5 (Other Race - Including Multi-Racial).
# ## Códigos 7 e 9 são considerados NAN e também serão unidos em cada coluna DPQx0.
# ## Na coluna 'INDFMINC' foram separados por categorias (faixa salarial por ano)

# ## O replace_map são informações que serão utilizadas para realizar as alterações mencionadas acima.

# In[ ]:


replace_map={
    'DPQ010':{7:np.nan, 9:np.nan},
    'DPQ020':{7:np.nan, 9:np.nan},
    'DPQ030':{7:np.nan, 9:np.nan},
    'DPQ040':{7:np.nan, 9:np.nan},
    'DPQ050':{7:np.nan, 9:np.nan},
    'DPQ060':{7:np.nan, 9:np.nan},
    'DPQ070':{7:np.nan, 9:np.nan},
    'DPQ080':{7:np.nan, 9:np.nan},
    'DPQ090':{7:np.nan, 9:np.nan},
    'RIDRETH1':{5:2},
    'DMDEDUC':{7:np.nan, 9:np.nan},
    'IMDFMIC':{1:np.mean([0,4999]), 2:np.mean([5000,9999]),3:np.mean([10000,14999]),
               4:np.mean([15000,19999]),5:np.mean([20000,24999]),6:np.mean([25000,34999]),
               7:np.mean([35000,44999]), 8:np.mean([45000,54999]), 9:np.mean([55000,64999]),
               10:np.mean([65000, 74999]),11:np.mean([75000]), 12:np.mean([20000,90000]),
               13:np.mean([0, 19999]), 77:np.nan, 99:np.nan}
}


# ##Nesse momento deve-se incluir as alterações no banco, porém por medida de segurança essas aplicações serão feitas em um novo banco (db2). Dessa maneira mantem-se o banco original para futuras consultas.

# In[ ]:


db2 = db.replace(replace_map)


# ##Verificar se o banco foi criado com sucesso.

# In[ ]:


db2.head()


# ##Seguindo com a análise, agora no novo banco (db2), faz-se necessária a checagem em busca de dados faltantes. Importante essa busca, pois dados faltantes podem prejudicar os resultados parciais e finais.

# In[ ]:


db2[['DPQ010',
     'DPQ020',
     'DPQ030',
     'DPQ040',
     'DPQ050',
     'DPQ060',
     'DPQ070',
     'DPQ080',
     'DPQ090']].isnull().mean()


# ##O próximo passo seria construir a phq9, como solicitado no desafio. Para construir será somado os valores de cada coluna individualmente, o parâmetro (axis='columns') terá essa tarefa. 

# In[ ]:


db2['phq9'] = db2[['DPQ010',
                   'DPQ020',
                   'DPQ030',
                   'DPQ040',
                   'DPQ050',
                   'DPQ060',
                   'DPQ070',
                   'DPQ080',
                   'DPQ090']].sum(axis='columns', skipna=False)


# ##Checando se deu certo o comando

# In[ ]:


db2[["DPQ010", 
     "DPQ020", 
     "DPQ030", 
     "DPQ040",
     "DPQ050", 
     "DPQ060", 
     "DPQ070", 
     "DPQ080", 
     "DPQ090",
     "phq9"]].head()


# ##Seguindo com as orientações do desafio, faz-se necessária a criação da variável phq_grp. Essa variável armazenará os valores de acordo com as condições estabelecidas.

# In[ ]:


conditions = [
              (db2['phq9'].isna()),
              (db2['phq9'] <= 5),
              (db2['phq9'] >5) & (db2['phq9'] <= 9),
              (db2['phq9'] >9) & (db2['phq9'] <=14),
              (db2['phq9'] >14) & (db2['phq9'] <=19),
              (db2['phq9'] > 19)
              ]

values = [np.nan,0,1,2,3,4]

db2['phq9_grp'] = np.select(conditions, values) #Contruindo de acordo com o passado antes.
db2['phq9_grp'].value_counts(sort=False) #Checando se deu certo


# ##Com essa observação, concluiu-se que a soma dos valores das categorias 2,3 e 4 ajudariam na análise. Dessa maneira, o comando abaixo fará essa tarefa.

# In[ ]:


db2['phq_grp2'] = db2['phq9_grp'].replace([3,4], 2)
db2['phq_grp2'].value_counts(sort=False)


# 1) Para as etapas de análise exploratória e teste de hipótese, utiliza-se a variável phq_grp2 com 3 níveis de sintomas de depressão.
# 
# 2) Como o percentual de missing está abaixo de 10% para todas as variáveis, não será feito nenhum tratamento para os casos faltantes.
# 

# ###EDA: Análise Univariada

# In[ ]:


var_quant = [
    "RIDAGEYR", 
    "INDFMINC", 
    "PAG_MINW", 
    "HEI2015C1_TOTALVEG",
    "HEI2015C2_GREEN_AND_BEAN",
    "HEI2015C3_TOTALFRUIT",
    "HEI2015C4_WHOLEFRUIT",
    "HEI2015C5_WHOLEGRAIN",
    "HEI2015C6_TOTALDAIRY",
    "HEI2015C7_TOTPROT",
    "HEI2015C8_SEAPLANT_PROT",
    "HEI2015C9_FATTYACID",
    "HEI2015C10_SODIUM",
    "HEI2015C11_REFINEDGRAIN",
    "HEI2015C12_SFAT",
    "HEI2015C13_ADDSUG",
    "HEI2015_TOTAL_SCORE",
    "phq9"]

var_quali = [
    "RIAGENDR",
    "RIDRETH1",
    "DMDEDUC",
    "ADHERENCE",
    "phq_grp2"
]

label_qual = {
    'RIAGENDR': {1: 'Masculino', 2: 'Feminino'},
    'RIDRETH1': {1: 'Americano Mexicano', 2: 'Outro', 3: 'Branco \n não hispânico', 4: 'Negro \n não hispânico'},
    'DMDEDUC': {1: '< 0 ano', 2: '0-12 Ano', 3: 'Ensino \n médio', 4: 'Superior \n incompleto', 5: 'Superior \n Completo'},
    'ADHERENCE': {1: 'Baixo', 2: 'Adequado', 3: 'Alto'},
    'phq_grp2': {0: 'Sem sintomas', 1: 'Sintomas \n leves', 2: 'Sintomas \n moderados-severos'}
}


# In[ ]:





# In[ ]:


db2[var_quali].describe(percentiles = [.25, .5, .75, .95, .99]).round(2)


# In[ ]:


db2[var_quant].describe(percentiles = [.25, .5, .75, .95, .99]).round(2)


# ## Observando a coluna que corresponde a minutos de exercicios, notou-se um problema, há registros que ultrapassam a quantidade de minutos por semana o que está visiviomente errado, para contornar, decidiu-se truncar o valor em 3600. Ou seja, uma semana tem 3600 minutos, agora os valores estão limitados a uma semnana.

# In[ ]:


db2['PAG_MINW_trunc'] = np.where(db2['PAG_MINW'] > 3600, 3600, db2['PAG_MINW'])


# In[ ]:


db2[['PAG_MINW', 'PAG_MINW_trunc']].describe([.25, .5, .75, .95, .99]).round(2)


# ##Para facilitar a leitura dos dados, eles serão convertidos em horas por semana

# In[ ]:


db2['PAG_MINW'] = db2['PAG_MINW_trunc'] / 60


# ##Preparando os gráficos...

# In[ ]:


sns.displot(db2, x='RIDAGEYR', kde=True)
sns.displot(db2, x='INDFMINC', kde=True)
sns.displot(db2, x='PAG_MINW', kde=True)
sns.displot(db2, x='PAG_MINW_trunc', kde=True)


# ##Com o objetivo de deixar o gráfico menos assimétrico, criou-se um gráfico em log dos valores de 'PAG_MINW'

# In[ ]:


db2['PAG_MINW_log'] = np.log(db2['PAG_MINW']+1)


# In[ ]:


sns.displot(db2, x='PAG_MINW_log', kde=True)


# ##Nota-se uma melhora na dstribuição dos valores.

# ## Nesse passo verifca-se a frequência dos valores nas variáveis qualitativas...

# In[ ]:


db2[var_quali].agg('value_counts').round(0)


# ##Algumas proporções também podem ser observadas

# In[ ]:


# Proporções
db2[var_quali].agg(pd.Series.value_counts, normalize=True).round(2)

# RIDRETH1 com códigos errados no desafio
# 1	Mexican American	
# 2	Other Hispanic
# 3	Non-Hispanic White	
# 4	Non-Hispanic Black	
# 5	Other Race - Including Multi-Racial


# ## Apresentação de algumas análises em gráficos

# ## Primeiro, recomenda-se a construção de uma função, ela terá a tarefa de criar os gráficos a partir de informações passadas pelo cientista.

# In[ ]:


# Função para construir gráfico de barras

def grafico_barras_prop(data, variable, values, label):
    (data[[variable]]
     .replace(values, label)
     .value_counts(normalize=True, sort = False)
     .rename("Proportion")
     .reset_index()
     .pipe((sns.barplot, "data"), x=variable, y="Proportion", order = label))
    plt.ylim(0,1)
    plt.show()


# In[ ]:


grafico_barras_prop(db2, variable = "RIAGENDR", values = [1,2], label = ["Masculino", "Feminino"])


# In[ ]:


grafico_barras_prop(db2, 
                    variable = "RIDRETH1", 
                    values = [1,2,3,4], 
                    label = ["Americano Mexicano", "Outro", "Branco não hispânico", "Negro não hispânico"])


# In[ ]:


grafico_barras_prop(db2, 
                    variable = "DMDEDUC", 
                    values = [1,2,3,4,5], 
                    label = ["< 9 ano", "9-12 ano", "Ensino médio", "Sup. incompleto", "Sup. completo"])


# In[ ]:


grafico_barras_prop(db2, 
                    variable = "ADHERENCE", 
                    values = [1,2,3], 
                    label = ["Abaixo", "Adequado", "Acima"])


# In[ ]:


grafico_barras_prop(db2, 
                    variable = "phq_grp2", 
                    values = [0,1,2], 
                    label = ["Sem sintomas", "Sintomas leves", "Sintomas moderados-severos"])


# ##No display é exibido todas as variáveis quantitativas e seus números

# In[ ]:


from IPython.display import display

for var in var_quant:
    display(db2[['phq_grp2', var]].groupby('phq_grp2').describe().round(2))


# ## EDA: Análise Bivariada

# ### A partir desse ponto, a análise será bivariada, ou seja, como uma variável se comporta se comparada com outra.

# In[ ]:


# Função para construir o boxplot

def grafico_boxplot_grp(data, variable, label):
  if label == "": label = variable
  sns.boxplot(x="phq_grp2", y=variable, data=data)
  plt.ylabel(label)
  plt.show()


# ### Observando os gráficos

# In[ ]:


grafico_boxplot_grp(db2.replace(label_qual),"RIDAGEYR",'Idade')


# In[ ]:


grafico_boxplot_grp(db2.replace(label_qual),"INDFMINC",'Renda Anual Familiar US$')


# In[ ]:


grafico_boxplot_grp(db2.replace(label_qual),"PAG_MINW",'Atividade Física (min/semana)')


# In[ ]:


grafico_boxplot_grp(db2.replace(label_qual),"HEI2015_TOTAL_SCORE",'HEI - Escore total')


# ##B)Perfil de hábitos saudáveis

# Alimentação saudável x Exercícios Físicos

# In[ ]:


from IPython.core.pylabtools import figsize
sns.boxplot(y='ADHERENCE',
            x='HEI2015_TOTAL_SCORE',
            orient='h',
            data=db2.replace(label_qual))
plt.show()

fig, ax = plt.subplots(ncols=2, figsize=(15,5))

sns.regplot(x='HEI2015_TOTAL_SCORE',
            y='PAG_MINW',
            lowess=True,
            line_kws={'color': 'red'},
            data = db2,
            ax = ax[0])

sns.regplot(x='HEI2015_TOTAL_SCORE',
            y='PAG_MINW_log',
            lowess=True,
            line_kws={'color': 'red'},
            data = db2,
            ax = ax[1])


# Hábitos saudáveis X Gênero

# In[ ]:


plt.figure(figsize=(7,5))
sns.boxplot(x = 'RIAGENDR',
            y = 'HEI2015_TOTAL_SCORE',
            data = db2.replace(label_qual))
plt.show()

fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

sns.boxplot(x = 'RIAGENDR',
            y = 'PAG_MINW',
            data = db2.replace(label_qual),
            ax = ax[0])

sns.boxplot(x = 'RIAGENDR',
            y = 'PAG_MINW_log',
            data = db2.replace(label_qual),
            ax = ax[1])
plt.show()


# Hábitos saudáveis X Etnias

# In[ ]:


plt.figure(figsize=(7,5))
sns.boxplot(x = 'RIDRETH1',
            y = 'HEI2015_TOTAL_SCORE',
            data = db2.replace(label_qual))
plt.show()

fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

sns.boxplot(x = 'RIDRETH1',
            y = 'PAG_MINW',
            data = db2.replace(label_qual),
            ax = ax[0])

sns.boxplot(x = 'RIDRETH1',
            y = 'PAG_MINW_log',
            data = db2.replace(label_qual),
            ax = ax[1])
plt.show()


# Hábitos saudáveis X Idade

# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(20,2))

sns.regplot(x='RIDAGEYR',
            y='HEI2015_TOTAL_SCORE',
            lowess=True,
            line_kws={'color': 'red'},
            data = db2,
            ax = ax[0])

sns.boxplot(x='RIDAGEYR',
            y='ADHERENCE',
            orient = 'h',
            data = db2.replace(label_qual),
            ax = ax[1])


# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(20,5))

sns.regplot(x='RIDAGEYR',
            y='PAG_MINW',
            lowess=True,
            line_kws={'color': 'red'},
            data = db2,
            ax = ax[0])

sns.regplot(x='RIDAGEYR',
            y='PAG_MINW_log',
            lowess=True,
            line_kws={'color': 'red'},
            data = db2,
            ax = ax[1])


# Hábitos saudávies X Renda

# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(20,5))

sns.regplot(x='INDFMINC',
            y='HEI2015_TOTAL_SCORE',
            lowess=True,
            line_kws={'color': 'red'},
            data = db2,
            ax = ax[0])

sns.boxplot(y='ADHERENCE',
            x='INDFMINC',
            orient = 'h',
            data = db2.replace(label_qual),
            ax = ax[1])


# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(20,5))

sns.regplot(x='INDFMINC',
            y='PAG_MINW',
            lowess=True,
            line_kws={'color': 'red'},
            data = db2,
            ax = ax[0])

sns.regplot(x='INDFMINC',
            y='PAG_MINW_log',
            lowess=True,
            line_kws={'color': 'red'},
            data = db2,
            ax = ax[1])


# ## TESTES DE HIPÓTESES

# In[ ]:


db3 = db2[['phq_grp2', 'PAG_MINW_log', 'HEI2015_TOTAL_SCORE']].dropna()


# In[ ]:


grafico_boxplot_grp(db2.replace(label_qual), 'HEI2015_TOTAL_SCORE', 'HEI - ESCORE TOTAL')


# In[ ]:


from scipy.stats import f_oneway


# In[ ]:


stat, p = f_oneway (db3[(db3.phq_grp2 == 0)] ['HEI2015_TOTAL_SCORE'],
                    db3[(db3.phq_grp2 == 1)] ['HEI2015_TOTAL_SCORE'],
                    db3[(db3.phq_grp2 == 2)] ['HEI2015_TOTAL_SCORE'])
print('stat = %.3f, p = %.3f', (stat, p))


# In[ ]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[ ]:


tukey = pairwise_tukeyhsd(db3['HEI2015_TOTAL_SCORE'], db3['phq_grp2'], alpha = 0.05)
print(tukey)


# ##Existe diferença no tempo médio de exercícios de acordo com grupos de depressão?

# In[ ]:


grafico_boxplot_grp(db2.replace(label_qual), 'PAG_MINW_log', 'PAG_MINW_log')


# In[ ]:


from scipy.stats import f_oneway


# In[ ]:


stat, p = f_oneway (db3[(db3.phq_grp2 == 0)] ['PAG_MINW_log'],
                    db3[(db3.phq_grp2 == 1)] ['PAG_MINW_log'],
                    db3[(db3.phq_grp2 == 2)] ['PAG_MINW_log'])
print('stat = %.3f, p = %.3f', (stat, p))


# In[ ]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[ ]:


tukey = pairwise_tukeyhsd(db3['PAG_MINW_log'], db3['phq_grp2'], alpha = 0.05)
print(tukey)

