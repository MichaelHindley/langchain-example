import uuid
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
import pinecone
import tiktoken

OPENAI_API_KEY = ""  # platform.openai.com
PINECONE_API_KEY = ""  # app.pinecone.io
PINECONE_ENV = "northamerica-northeast1-gcp"

tokenizer = tiktoken.get_encoding('cl100k_base')  # cl100k base is encoder used by ada-002
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
index_name = "fridman"

pinecone.init(
    api_key=PINECONE_API_KEY,  # app.pinecone.io
    environment=PINECONE_ENV  # next to API key in console
)

index = pinecone.Index(index_name)
batch_size = 100
       
# create uuidv4 ids for each record (this is required by Pinecone)

# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         index_name, dimension=1536
#     )

def main():
    print("Starting")
    # createIndex()
    createScaniaSustainabilityIndex()
    scaniaChat()
    # pokemonChat()
    # createPokemonIndex()
    # chat()


def createScaniaSustainabilityIndex():
    data = load_dataset(
        './txt/',
        split='train'
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,  # number of tokens overlap between chunks
        length_function=tiktoken_len,  # token count function
        separators=['\n\n', '.\n', '\n', '.', '?', '!', ' ', '']
    )

    new_data = []
    for row in tqdm(data):
        chunks = text_splitter.split_text(row['text'])
        for i, text in enumerate(chunks):
            new_data.append({**row, **{'chunk': i, 'text': text}})

    for i in tqdm(range(0, len(new_data), batch_size)):
        # get end of batch
        i_end = min(len(new_data), i+batch_size)
        # get batch of records
        metadatas = new_data[i:i_end]
        ids = [f"{uuid.uuid4()}-{meta['chunk']}" for meta in metadatas]
        texts = [meta['text'] for meta in metadatas]
        xc = embeddings.embed_documents(texts)
        to_upsert = zip(ids, xc, metadatas)
        # now add to Pinecone vec DB
        index.upsert(vectors=to_upsert)

def scaniaChat():
    vectordb = Pinecone(
        index=index,
        embedding_function=embeddings.embed_query,
        text_key="text"
    )

    llm=ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        model_name='gpt-3.5-turbo'
    )

    retriever = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    tool_desc = """Use this tool to answer user questions using Scania sustainability. If the user states 'ask scania' use this tool to get
    the answer. This tool can also be used for follow up questions from
    the user."""

    tools = [Tool(
        func=retriever.run,
        description=tool_desc,
        name='Scania Sustainability Report DB'
    )]

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",  # important to align with agent prompt (below)
        k=5,
        return_messages=True
    )

    conversational_agent = initialize_agent(
        agent='chat-conversational-react-description', 
        tools=tools, 
        llm=llm,
        verbose=True,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory,
    )

    sys_msg = """You are a helpful chatbot that answers the user's questions about Scanias sustainability report.
"""

    prompt = conversational_agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    conversational_agent.agent.llm_chain.prompt = prompt

    conversational_agent("ask scania what are some differences between scanias sustainability report from 2020 and scanias sustainability report from 2022")
    

def pokemonChat():
    vectordb = Pinecone(
        index=index,
        embedding_function=embeddings.embed_query,
        text_key="text"
    )

    llm=ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        model_name='gpt-3.5-turbo'
    )

    retriever = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    tool_desc = """Use this tool to answer user questions using a Pokedex. If the user states 'ask pokedex' use this tool to get
    the answer. This tool can also be used for follow up questions from
    the user."""

    tools = [Tool(
        func=retriever.run,
        description=tool_desc,
        name='Pokedex DB'
    )]

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",  # important to align with agent prompt (below)
        k=5,
        return_messages=True
    )

    conversational_agent = initialize_agent(
        agent='chat-conversational-react-description', 
        tools=tools, 
        llm=llm,
        verbose=True,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory,
    )

    sys_msg = """You are a helpful chatbot that answers the user's questions.
"""

    prompt = conversational_agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    conversational_agent.agent.llm_chain.prompt = prompt

    conversational_agent("ask pokedex what is the description of bulbasaur")


def chat():
    vectordb = Pinecone(
        index=index,
        embedding_function=embeddings.embed_query,
        text_key="text"
    )

    llm=ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        model_name='gpt-3.5-turbo'
    )

    retriever = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    tool_desc = """Use this tool to answer user questions using Lex
    Fridman podcasts. If the user states 'ask Lex' use this tool to get
    the answer. This tool can also be used for follow up questions from
    the user."""

    tools = [Tool(
        func=retriever.run,
        description=tool_desc,
        name='Lex Fridman DB'
    )]

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",  # important to align with agent prompt (below)
        k=5,
        return_messages=True
    )

    conversational_agent = initialize_agent(
        agent='chat-conversational-react-description', 
        tools=tools, 
        llm=llm,
        verbose=True,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory,
    )

    sys_msg = """You are a helpful chatbot that answers the user's questions.
"""

    prompt = conversational_agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    conversational_agent.agent.llm_chain.prompt = prompt

    conversational_agent("ask lex about the future of ai")

def createPokemonIndex():
    data = load_dataset(
        'mfumanelli/pokemon-description-xs',
        split='train'
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,  # number of tokens overlap between chunks
        length_function=tiktoken_len,  # token count function
        separators=['\n\n', '.\n', '\n', '.', '?', '!', ' ', '']
    )

    new_data = []
    for row in tqdm(data):
        chunks = text_splitter.split_text(row['description'])
        row.pop('description')
        for i, text in enumerate(chunks):
            new_data.append({**row, **{'chunk': i, 'text': text}})

    for i in tqdm(range(0, len(new_data), batch_size)):
        # get end of batch
        i_end = min(len(new_data), i+batch_size)
        # get batch of records
        metadatas = new_data[i:i_end]
        ids = [f"{meta['name']}-{meta['chunk']}" for meta in metadatas]
        texts = [meta['text'] for meta in metadatas]
        xc = embeddings.embed_documents(texts)
        to_upsert = zip(ids, xc, metadatas)
        # now add to Pinecone vec DB
        index.upsert(vectors=to_upsert)


def createIndex():
    data = load_dataset(
        'jamescalam/lex-transcripts',
        split='train'
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,  # number of tokens overlap between chunks
        length_function=tiktoken_len,  # token count function
        separators=['\n\n', '.\n', '\n', '.', '?', '!', ' ', '']
    )

    new_data = []
    for row in tqdm(data):
        chunks = text_splitter.split_text(row['transcript'])
        row.pop('transcript')
        for i, text in enumerate(chunks):
            new_data.append({**row, **{'chunk': i, 'text': text}})

    for i in tqdm(range(0, len(new_data), batch_size)):
        # get end of batch
        i_end = min(len(new_data), i+batch_size)
        # get batch of records
        metadatas = new_data[i:i_end]
        ids = [f"{meta['video_id']}-{meta['chunk']}" for meta in metadatas]
        texts = [meta['text'] for meta in metadatas]
        xc = embeddings.embed_documents(texts)
        to_upsert = zip(ids, xc, metadatas)
        # now add to Pinecone vec DB
        index.upsert(vectors=to_upsert)

# define a length function
def tiktoken_len(text: str) -> int:
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# start program
main()
