from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

genai.configure(api_key="AIzaSyA4pBHjWR1JQFuU5JqCLsZBv7gEz7La3kM")
os.environ["GOOGLE_API_KEY"] = "AIzaSyA4pBHjWR1JQFuU5JqCLsZBv7gEz7La3kM"

def get_model_response(file,query):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    context="\n\n".join(str(p.page_content) for p in file)
    data=text_splitter.split_text(context)
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    searcher=Chroma.from_texts(data,embeddings).as_retriever()

    q="Which Employee has maximum salary?"
    records=searcher.get_relevant_documents(q)
    print(records)


    prompt_template="""
    You have to answer the question from the provided context and make sure that you provided all the details \n
    Context :{context}?\n
    Question :{question}\n

    Answer


    """
    prompt=PromptTemplate(template=prompt_template,input_variables=["context", "question"])

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.9)

    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)

    response=chain(
        {
            "input_documents":records,
            "question":query
        }
        ,return_only_outputs=True)
    
    return response['output_text']