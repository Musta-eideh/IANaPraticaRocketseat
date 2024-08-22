#   PROJETO REALIZADO NO 'GOOGLE COLAB'
# INSTALAÇÃO DAS LIBS
# !pip install yfinance==0.2.41

# !pip install crewai==0.28.8

# !pip install 'crewai[tools]'

# !pip install langchain-openai==0.1.7 

# !pip install langchain-community==0.0.38

# !pip install duckduckgo-search==5.3.0

# IMPORT DAS LIBS
import json
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

from IPython.display import Markdown

# FAZENDO DOWNLOAD DA COTAÇÃO DO TICKET DA APPLE
## com inicio e fim de prazos para se capturar
stockApple= yf.download("AAPL", start="2023-08-08", end="2024-08-08")
print(stockApple)

#CRIANDO YAHOO FINANCE TOOL
def fetchStockPrice(ticket):
  stockApple = yf.download("AAPL", start="2023-08-08", end="2024-08-08")
  return stockApple

yahooFinanceTools = Tool(
    name = "Yahoo Finance Tool",
    description = "Fecthes stock prices for {ticket} from the last year about specific stock from Yahoo Finance API",
    func= lambda ticket: fetchStockPrice(ticket)
)

# Verificando se está funcionando nossa funcionalidade de demonstrar todas as cotações feitas do dia selecionado
response = yahooFinanceTools.run("AAPL")
print(response)

# Agora de fato usaremos o Agent AI
# IMPORTANDO OPEN LLM - GPT
os.environ['OPEN_API_KEY'] = "PLEASE-ENTER-YOUR-TOCKEN-HEAR"
llm = ChatOpenAI(model="gpt-3.5-turbo")
# u need have payment with GPT

# Iremos criar o próprio Agent
stockPriceAnalyst = Agent(
    role = "Senior stock Price Analist",
    goal = "Find the {ticket} stock price and analyses trends",
    backstory = """You`re a highly experienced in analyzing the price of an specific 
    stock and make predictions its future price.""",
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    tools = [yahooFinanceTools],
    allow_delegation = False
)

# CRIANDO TAREFA PARA SER EXECUTADA
getStockPrice = Task(
    description = "Analyse the stock {ticket} price history and create a trend analyses  of up, down or sideways",
    expected_output = """ Specify the current trend stock price - up, down or sideway. eg. stock='APPL' price Ip """,
    agent = stockPriceAnalyst
)

## Então até aqui criamos nosso primeiro agent que usa o yahoo finance tool e que tem a task de pegar o preço da ação

# IMPORTANDO A TOOL DE SEARCH
searchTool = DuckDuckGoSearchResults(backend='news', num_results=10)

# AGORA IREMOS CRIAR O AGENTE DE ANALISTA
newsAnalyst = Agent(
    role = "Stock news analyst",
    goal = """Create a short summary of the market news related to the stock {ticket} company. Specify the current 
     trend - up, down or sideways with the news context. For eat request stock asset, specify a number of
     between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory = """ You`re highly experienced in analyzin the market trends and news and have tracked assets for more
    the 10 years. 
    
    You`re also master level analyst in the tradicional markets and have deep understanding of human psychology.    
    
    You understand news, theirs tittles and information, but you look at those with a health dose of skepticism.
    You consider also the source of the new articles.
    """,
    verbose = True,
    llm = llm,
    max_iter = 10,
    memory = True,
    tools = [searchTool],
    allow_delegation = False
)

# CRIANDO A TASK DO AGENTE DE ANALISTA
getNews = Task(
    description = """Take the stock and always include BTC to it (if not request).
    Use the search tool to search each one individually.

    The current date is {datetime.now()}.

    Compose the results into a helpfull report.
    """,
    expected_output = """ A summary of the overeall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent = newsAnalyst
)

# CRIAREMOS O AGENTE QUE IRÁ FAZER/ESCREVER A ANALISE DE FATO
stockAnalystWriter = Agent(
    role = "Senior analyst Writer",
    goal = """ Alayse the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price
    trend. 
    """,
    backstory = """ You`re widely accepted as the best stock analyst in the marcket.
    You undertand complex concepts and create compelling stories and naratives that resonate with
    wider audiences.

    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. You`re able to
    hold multiple opinions when analyzing anything.
    """,
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    allow_delegation = True
)

# NOSSO ANALISTA TAMBÉM PRECISA DE UMA TASK
writeAnalyses = Task(
    description = """  Use the stock price trend and the stock news report to create an analyses and 
    write the newsletter about the {ticket} company that is brieg and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?

    Include the previous analyses of stock trend and news summary.
    """,
    expected_output = """ An eloquent 3 paragraphs newsletter formated as markdown in an easy readable
    manner. It should contain:
    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analyses including the news summary and fead/greed scores
    - summary - key facts and concreate future trend prediction - up, down or sideways.
    """,
    agent = stockAnalystWriter,
    context = [getStockPrice, getNews]
)

# AGORA IREMOS CRIAR O NOSSO GRUPO DE AGENTES DE IA
crew = Crew(
    agent = [stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks = [getStockPrice, getNews, writeAnalyses],
    verbose = 2,
    process = Process.hierarchical ,
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iters = 15
)

# AGORA IREMOS EXECUTAR NOSSA CREW 
results = crew.kickoff(Inputs={'ticket':'AAPL'})

# VERIFICAÇÕES
list(results.keys())  # O QUE VEM NO RESULTS - DOIS PARAMETROS
results['final_output'] # UM DOS PARAMETROS - VEMOS O QUE O FINAL OUTPUT TRAGO
len(results['tasks_outputs']) # POSSUI 3 TAKS QUE CADA UMA GEROU UM RESULTADO

# AGORA IREMOS FAZER COM QUE FIQUE BOM PARA LEITURA
Markdown(results['final_output'])

# CONVERTE O ARQUIVO
# prompt: jupyter nbconvert --to script crewai-stocks.ipnb

# PODE REMOVER O MARKDOWN LOGO EM SEGUIDA, PORQUE NÃO UTILZIAREMOS
# POR ENQUANTO DEIXAREMOS A CHAVE DE API, MAS TIRAREMOS ELA EM BREVE

# LOGO EM SEGUIDA PODE-SE IMPORTAR UMA NOVA BIBLIOTECA (STREAMLIT)
import streamlit as st
# prompt: pip install streamlit
# O QUE O STREAMLIT FORNECE? FORNECE UM MÉTODO PARA CONSTRUIRMOS UMA APLICAÇÃO WEB
# DE PYTHON DE UMA MANEIRA MUITO RÁPIDA

# COMO EXECUTAR COM O STREAMLIT TODA ESSA ARQUITETURA DE CÓDIGO
# AO FINAL DO CÓDIGO

with st.sidebar:
  st.header('Enter the Stock to Research')

  with st.form(key='research_form'):
    topic = st.textInput("Select the ticket")
    submitBtn = st.formSubmitBtn(label="Run Research")

if submitBtn:
  if not topic:
    st.error("Plesase fill the ticket field")
  else:
    results = crew.kickoff(inputs={'ticket': topic })

    st.subheader("Results of your research")
    st.write(results['finalOutput']
)
    
## PARA RODAR PRECISARÁ DO SEGUINTE COMANDO NO PROMPT:
# PROMPT: streamlit run crewai-stocks.py 