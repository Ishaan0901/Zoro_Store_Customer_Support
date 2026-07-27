from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Customer Care", page_icon="🤖")
st.title("Welcome to Zoro Store's Customer Support Chatbot !!")
st.caption('Firstly we are sorry that u have to come here because of the difficulty in our Product !')

#initializing chat history for UI
if "messages" not in st.session_state:
    st.session_state.messages = []

#Converting the history to text
def get_chat_history():
    history = ""

    for message in st.session_state.messages:
        history += (
            f"{message['role'].capitalize()}: "
            f"{message['content']}\n"
        )

    return history



@st.cache_resource
def load_chain():

    #Template
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a customer support agent at Zoro Store.

    Your job is to answer the customer's query ONLY using the provided context and the previous conversation.

    Instructions:
    - Read the customer's issue carefully.
    - Use the previous conversation only to understand follow-up questions.
    - Use the provided context to answer the query.
    - Do NOT make up information.
    - If the context does not contain the answer, reply exactly:

    "I'm sorry! I am unable to answer your query based on the available information. If you'd like, I can forward your issue to our customer support team."

    Keep your responses polite, professional, and concise.
    """
        ),
        (
            "human",
            """Previous Conversation:
    {history}

    Customer Issue:
    {issue}

    Retrieved Context:
    {context}"""
        )
    ]
    )

    #Embedding model:
    emb_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True,
    }
)

    #ChatModel:
    chat_model=ChatMistralAI(
        model_name='mistral-small-2603'
    )

    #Retriever:
    vectorstore=Chroma(
        persist_directory='zoro_store_database',
        embedding_function=emb_model
    )

    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
        "k":3
        }
    )

    #function to join the output of the retriever:
    def format_docs(docs):
        formatted_docs = "\n\n".join(doc.page_content for doc in docs)
        return formatted_docs

    #parser:
    parser=StrOutputParser()

    #chain:
    flow = RunnableParallel(
    {
        "issue": RunnableLambda(lambda x: x["issue"]),
        "history": RunnableLambda(lambda x: x["history"]),
        "context": RunnableLambda(lambda x: x["issue"])
                   | retriever
                   | RunnableLambda(format_docs)
    }
    ) | prompt | chat_model | parser


    return flow


#Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query=st.chat_input('Tell me the issue you are facing...')

with st.status("🤖 Preparing chatbot...", expanded=False) as status:
    chain = load_chain()
    status.update(label="✅ Chatbot is ready!", state="complete")

if query:
    
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("🤖 Thinking..."):
        history = get_chat_history()
        result = chain.invoke(
            {
                "issue": query,
                "history": history
            }
        )

    st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result
        }
    )
    with st.chat_message("assistant"):
        st.write(result)