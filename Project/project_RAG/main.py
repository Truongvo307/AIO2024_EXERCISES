import torch

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

Loader = PyPDFLoader
FILE_PATH = "./YOLOv10_Tutorials.pdf"

# Create an instance of the loader with the specified file path
loader = Loader(FILE_PATH)

# Load the documents
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter (chunk_size =1000,chunk_overlap =100)
docs = text_splitter.split_documents(documents)
print("Number of sub - documents : ", len (docs))
print(docs[0])

# Convert the documents to vectors
# Initialize the embedding model
embedding = HuggingFaceEmbeddings()

# Create the Chroma vector database from documents
vector_db = Chroma.from_documents(documents=documents, embedding=embedding)

# Create a retriever from the vector database
retriever = vector_db.as_retriever()

result = retriever.invoke(" What is YOLO ?")
print ("Number of relevant documents : ", len (result))

#LLM model (Vivuna)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Initialize the model
MODEL_NAME = "lmsys/vicuna-7b-v1.5"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#Combine tokenizaer and model to create a pipeline
model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto"
)

llm = HuggingFacePipeline (pipeline = model_pipeline)


#Run the model
prompt = hub.pull("rlm/rag-prompt")

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# User question
USER_QUESTION = "YOLOv10 là gì?"

# Invoke the RAG chain with the user question
output = rag_chain.invoke(USER_QUESTION)

# Extract and print the answer
answer = output.split('Answer:')[1].strip()
print(answer)