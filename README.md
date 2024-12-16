# Seleção de Alvos Terapêuticos com LLMs

A seleção de alvos é uma etapa essencial nos estágios iniciais da descoberta de fármacos. A identificação de alvos terapêuticos é um processo complexo e demorado, que envolve a análise de uma grande quantidade de dados. A aplicação de grandes modelos de linguagem (LLMs) pode acelerar esse processo, permitindo a identificação de alvos terapêuticos potenciais com base em abstracts de artigos científicos disponíveis no PubMed.

Neste projeto, utilizamos uma estrutura de Retrieval Augmented Generation (RAG) para selecionar alvos terapêuticos com base em abstracts de artigos científicos. O modelo RAG utiliza (1) um modelo de geração de embeddings e um banco de dados vetorial para recuperar documentos relevantes e (2) um modelo de geração de linguagem para extrair as informações de alvos terapêuticos dos documentos recuperados.

## Instruções

Para executar o código, é necessário possuir Python e Docker instalados no seu sistema. O banco de dados utilizado é o PostgreSQL com a extensão `pgvector`, instanciado a partir de um container Docker.

A aplicação espera que as variáveis de ambiente `DATABASE_URL` e `GEMNI_API_KEY` estejam definidas. A primeira variável de ambiente é a URL de conexão com o banco de dados, enquanto a segunda é a chave de acesso à API do Gemini.

1. Clone o repositório:

```bash
$ git clone https://github.com/arthurcerveira/Therapeutic-Target-Selection.git
$ cd Therapeutic-Target-Selection/
```

2. Instale as dependências:

```bash
$ pip install -r requirements.txt
```

3. Baixe os dados do PubMed:

```bash
$ cd data/
$ python download_pubmed.py
```

4. Instanciar o banco de dados:

```bash
$ docker-compose up -d
```

5. Indexar os abstracts no banco de dados:

```bash
$ python load_abstracts_to_db.py
```

6. Executar o script principal. Para isso, é necessário possuir uma variável de ambiente `DATABASE_URL` com a URL de conexão com o banco de dados. Por exemplo:

```bash
$ python main.py
```

Este script irá instanciar uma aplicação Gradio (em http://127.0.0.1:7860) que permite a inserção de uma query com a doença considerada e, a partir dos abstracts recuperados, retorna os alvos terapêuticos relacionados.