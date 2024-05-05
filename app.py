from flask import Flask, request, render_template
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint


app=Flask(__name__,template_folder='templates')

@app.route('/')
def login():
    setup()
    return render_template('index2.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text

def setup():
    print("setting up ")
    OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]
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

    llm = OctoAIEndpoint(
        model="llama-2-13b-chat-fp16",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
        
    )
    embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")

    ZILLIZ_API_TOKEN = os.environ["ZILLIZ_API_TOKEN"]
    ZILLIZ_ENDPOINT = os.environ["ZILLIZ_ENDPOINT"]

    # vector_store = Milvus.from_documents(
    #     splits,
    #     embedding=embeddings,
    #     connection_args={"uri": ZILLIZ_ENDPOINT, "token": ZILLIZ_API_TOKEN},
    # collection_name="starwars"
    # )
    return ZILLIZ_ENDPOINT
