import os
import zipfile
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Acrux Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Acrux Dynamics HR Help Desk")
st.caption("Ask me anything about HR policies — leaves, salary, WFH, performance, and more!")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_pipeline():
    # Unzip hr_docs if not already extracted
    if not os.path.exists("hr_docs") or len(os.listdir("hr_docs")) == 0:
        with zipfile.ZipFile("hr_docs.zip", "r") as z:
            z.extractall("hr_docs")

    loader = PyPDFDirectoryLoader("hr_docs/")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
        api_key=GROQ_API_KEY
    )

    return retriever, llm

retriever, llm = build_pipeline()

REFUSAL_MESSAGE = "I'm sorry, I can only answer HR-related questions based on Acrux Dynamics policy documents. Please contact hr@acruxdynamics.com for further assistance."

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an HR Help Desk assistant for Acrux Dynamics Pvt. Ltd.
Answer ONLY using the provided context from official HR policy documents.
If the answer is not found in the context, respond with exactly:
"I'm sorry, I can only answer HR-related questions based on Acrux Dynamics policy documents. Please contact hr@acruxdynamics.com for further assistance."
Be concise, professional, and mention the relevant policy when possible.

Context:
{context}"""),
    ("human", "{question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a classifier. Determine if the question is related to HR policies,
employee benefits, leave, salary, performance, work from home, code of conduct,
onboarding, separation, travel expenses, or IT policies of a company.
Reply with only one word: YES if it is HR-related, NO if it is not."""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', 'Unknown')}]\n{d.page_content}"
        for d in docs
    )

def ask_bot(question):
    oos_check = OOS_PROMPT.format_messages(question=question)
    oos_response = llm.invoke(oos_check).content.strip().upper()
    if "NO" in oos_response:
        return {"answer": REFUSAL_MESSAGE, "sources": []}
    docs = retriever.invoke(question)
    context = format_docs(docs)
    prompt = RAG_PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(prompt)
    sources = list({d.metadata.get("source", "Unknown") for d in docs})
    return {"answer": response.content, "sources": sources}

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.write(f"• {os.path.basename(s)}")

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask_bot(prompt)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📄 Sources"):
                for s in result["sources"]:
                    st.write(f"• {os.path.basename(s)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", [])
    })