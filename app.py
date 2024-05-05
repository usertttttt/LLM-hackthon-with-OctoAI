from flask import Flask, request, render_template
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint

app=Flask(__name__,template_folder='templates')

@app.route('/')
def login():
    setup()
    
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    output = call_octoai(processed_text)
    return output
    # TODO: return output in index.html

def setup():
    print("setting up ")
    OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

def call_octoai(processed_text):
    url = "https://en.wikipedia.org/wiki/Star_Wars"

    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
        ("div", "Divider")
    ]

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # for local file use html_splitter.split_text_from_file(<path_to_file>)
    html_header_splits = html_splitter.split_text_from_url(url)

    chunk_size = 1024
    chunk_overlap = 128
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split
    splits = text_splitter.split_documents(html_header_splits)


    llm = OctoAIEndpoint(
        model="llama-2-13b-chat-fp16",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
        
    )
    embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")

    # vector_store = Milvus.from_documents(
    # splits,
    # embedding=embeddings,
    # connection_args={"host": "localhost", "port": 19530},
    # collection_name="starwars"
    # )

    ZILLIZ_API_TOKEN = os.environ["ZILLIZ_API_TOKEN"]
    ZILLIZ_ENDPOINT = os.environ["ZILLIZ_ENDPOINT"]

    vector_store = Milvus.from_documents(
        splits,
        embedding=embeddings,
        connection_args={"uri": ZILLIZ_ENDPOINT, "token": ZILLIZ_API_TOKEN},
    collection_name="starwars"
    )


    retriever = vector_store.as_retriever()

    template="""You are a literary critic. You are given some context and asked to answer questions based on only that context.
    Question: {question} 
    Context: {context} 
    Answer:"""
    lit_crit_prompt = ChatPromptTemplate.from_template(template)

    lcchain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | lit_crit_prompt
    | llm
    | StrOutputParser()
    )
    output = lcchain.invoke(processed_text)
    # output = lcchain.invoke("What is the worst thing about Darth Vader's story line?")
    pprint(output)

    return output
