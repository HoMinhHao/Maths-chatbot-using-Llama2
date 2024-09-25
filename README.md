# Maths-chatbot-using-Llama2
# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/HoMinhHao/Maths-chatbot-using-Llama2.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -p mchatbot python=3.11 -y
```

```bash
source activate mchatbot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Llama
- Pinecone