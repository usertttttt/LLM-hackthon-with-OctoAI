# for desktop set up
This version uses Milvus through Docker Compose so you must have Docker installed to run this notebook (Milvus is spun up via docker compose up -d)
# ! pip install -qU pymilvus langchain sentence-transformers tiktoken octoai-sdk openai
# ! docker-compose up -d
# for window set environment
For anyone on Windows that is new to running Jupyter notebooks, I highly recommend using WSL. The following are the steps that you need to follow on a clean system to get setup with Python, VS Code, and WSL to run the tutorial.

Enable WSL in windows features
Install VS Code from MS store
Install WSL from MS store
Install Ubuntu 24.04 LTS from the MS store

run Ubuntu 24.04 LTS from the start menu

>> run the following commands in WSL console
sudo apt update && sudo apt upgrade
sudo apt install python-is-python3 python3-pip python3.12-venv git

mkdir aihack
cd aihack
python3 -m venv aihackvenv
source aihackvenv/bin/activate

>>> start your pip installs; while the following is downloading continue with the other steps
pip install pymilvus milvus langchain sentence-transformers tiktoken octoai-sdk openai lxml

download notebook and move/copy to WSL/home/[username]aihack

open new WSL console, run "cd aihack" and "source aihackvenv/bin/activate" to activate your environment

export OCTOAI_API_TOKEN=[your API key from OCtoAI]
code .

Install the following extentions in VS Code:
1. WSL
>> restart WSL <<< by closing it and then running "code ." again
2. Jupyter  (install in WSL)
3. Jupyter Keymap  (install in WSL)
4. Python Debugger  (install in WSL)
>> restart WSL <<< by closing it and then running "code ." again

Open the notebook in your VS Code session then
click on "Select Kernel" in the upper right corner
pick "Python Environments"
pick aihackenv

Attempt to run the first non-commented block in the code
Install ipykernel when prompted by VS Code

Now you are all set!!!!
