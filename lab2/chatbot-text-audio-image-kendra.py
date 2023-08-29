import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from typing import Any, Dict, List, Optional
import json
from io import StringIO, BytesIO
from random import randint
from PIL import Image
import boto3
import numpy as np
import pandas as pd
import json
import os
import base64
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

client = boto3.client('runtime.sagemaker')
aws_region = boto3.Session().region_name
source = []
kendra_index_id = os.getenv("KENDRA_INDEX_ID", default="2d3b8bbd-40ae-4c47-9fc0-998d6e91de3b")
st.set_page_config(page_title="Document Analysis", page_icon=":robot:")


Falcon_endpoint_name = os.getenv("nlp_ep_name", default="falcon-7b-instruct-2xl")
CV_endpoint_name = os.getenv("cv_ep_name", default="blip2-flan-t5-xlarge")
whisper_endpoint_name = os.getenv('speech_ep_name', default="whisper-large-v2")
embedding_endpoint_name = os.getenv('embed_ep_name', default="huggingface-textembedding-gpt-j-6b-fp16-4xlarge")

endpoint_names = {
    "NLP":Falcon_endpoint_name,
    "Computer_Vision":CV_endpoint_name,
    "Audio":whisper_endpoint_name
}



################# Prepare for chatbot with memory #######################

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    len_prompt = 0

    def transform_input(self, prompt: str, model_kwargs: Dict={}) -> bytes:
        self.len_prompt = len(prompt)
        input_str = json.dumps({"inputs": prompt, "parameters":{"max_new_tokens": st.session_state.max_token, "temperature":st.session_state.temperature, "seed":st.session_state.seed, "stop": ["Human:"], "num_beams":1, "return_full_text": False, "repetition_penalty": 2.0}})
        print(input_str)
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = output.read()
        res = json.loads(response_json)
        print(res)
        ans = res[0]['generated_text']
        
        return ans[:ans.rfind("\nUser")].strip()


    
content_handler = ContentHandler()

llm = SagemakerEndpoint(
            endpoint_name=Falcon_endpoint_name,
            region_name="us-east-1",
            content_handler=content_handler,
    )

@st.cache_resource
def load_chain(endpoint_name: str=Falcon_endpoint_name):

    memory = ConversationBufferMemory(return_messages=True)
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

chatchain = load_chain()


# initialise session variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    chatchain.memory.clear()
if 'rag' not in st.session_state:
    st.session_state['rag'] = False
if 'widget_key' not in st.session_state:
    st.session_state['widget_key'] = str(randint(1000, 100000000))
if 'max_token' not in st.session_state:
    st.session_state.max_token = 200
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1
if 'seed' not in st.session_state:
    st.session_state.seed = 0
if 'extract_audio' not in st.session_state:
    st.session_state.extract_audio = False
if 'option' not in st.session_state:
    st.session_state.option = "NLP"
    
def clear_button_fn():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['widget_key'] = str(randint(1000, 100000000))
    st.widget_key = str(randint(1000, 100000000))
    st.session_state.extract_audio = False
    chatchain.memory.clear()
    # chatchain = load_chain(endpoint_name=endpoint_names['NLP'])
    # chatchain.memory.clear()
    uploaded_file=None
    st.session_state.option = "NLP"

    
    
def on_file_upload():
    st.session_state.extract_audio = True
    st.session_state['generated'] = []
    st.session_state['past'] = []
    # st.session_state['widget_key'] = str(randint(1000, 100000000))
    chatchain.memory.clear()
    
def prompt_hints(prompt_list):
    sample_prompt = []
    for prompt in prompt_list:
        sample_prompt.append( f"- {str(prompt)} \n")
    return ' '.join(sample_prompt)
        
prompts = {
                'rag':[
                    "Can SageMaker Asynchronous endpoint scale down to zero?",
                    "What is Amazon SageMaker Saving Plan?",
                    "what is the recommended way to first customize a foundation model?",
                    "if prompt engineering is not able to handle specific task, what approach can you use to handle domain-specific tasks?",
                    "how can Amazon SageMaker help with Machine Learning?",
                  ],
                'audio_prompt': [
                    "what does this file say? Summarize in one sentence",
                    "how can digital assets facilitate customers' engagement?"
                ],
                'image_prompt': [
                    "provide a caption of the image",
                    "how many cars do you see in the image?",
                    "did you see any people getting injured in the image?"
                ],
               'file_prompt': [
                   'what does this document talk about?',
                   'how much was the net sales?',
                   'based on the financial results, how you do see the future growth of Amazon'
               ],
                'default': [
                    "\n Based on the INPUT, answer the Question. INPUT TEXT: Canceling my banking and direct investing account to move to your competitor. You've lost a long time customer. \n\n QUESTION: what is the sentiment? \n\n OPTIONS: positive neutral negative. \n\n Helpful Answer:",
                    "write a kind message to the customer who provided the feedback"
                ]
              }
prompt_suggstion = prompt_hints(prompts['default'])

with st.sidebar:
    # Sidebar - the clear button is will flush the memory of the conversation
    st.sidebar.title("Conversation setup")
    clear_button = st.sidebar.button("Clear Conversation", key="clear", on_click=clear_button_fn)

    # upload file button
    uploaded_file = st.sidebar.file_uploader("Upload a file (text, image, or audio)", 
                                             key=st.session_state['widget_key'], 
                                             on_change=on_file_upload,
                                            )
    if uploaded_file:
        filename = uploaded_file.name
        print(filename)
        st.session_state.rag = False
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            st.session_state.option = "Computer_Vision"
            image = Image.open(uploaded_file)
            st.markdown('<p style="text-align: center;">uploaded image</p>',unsafe_allow_html=True)
            st.image(image,width=300)
            
            prompt_suggstion = prompt_hints(prompts['image_prompt'])
        elif filename.lower().endswith(('.flac', '.wav', '.webm', 'mp3')):
            st.session_state.option = "Audio"
            byteio = BytesIO(uploaded_file.getvalue())
            data = byteio.read()
            st.audio(data, format='audio/webm')
            prompt_suggstion = prompt_hints(prompts['audio_prompt'])
        else:
            st.session_state.option = "NLP"
            prompt_suggstion = prompt_hints(prompts['file_prompt'])
            
            
    rag = st.checkbox('Use knowledge base (answer question based on the retrieved relevant information from the video data source)', key="rag")

    if rag:
        prompt_suggstion = prompt_hints(prompts['rag'])
        retriever = AmazonKendraRetriever(index_id=kendra_index_id)
    st.sidebar.markdown(f'### Suggested prompts: \n\n {prompt_suggstion}')
            
    

    
left_column, _, right_column = st.columns([50, 2, 20])

with left_column:
    st.header("Building a multifunctional chatbot with Amazon SageMaker")
    # this is the container that displays the past conversation
    response_container = st.container()
    # this is the container with the input text box
    container = st.container()
    
    with container:
        # define the input text box
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Input text:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')
        
        
        # when the submit button is pressed we send the user query to the chatchain object and save the chat history
        if submit_button and user_input:
            if st.session_state.option == "Computer_Vision":
                payload = BytesIO(uploaded_file.getvalue()).read()
                data = { 
                    "image" : base64.b64encode(payload).decode(), 
                    "prompt": f"Question: {user_input}/n Answer:",
                    "parameters": { "max_length": 500 },
                }
                response = client.invoke_endpoint(EndpointName=endpoint_names["Computer_Vision"], 
                                                  ContentType='application/json', 
                                                  Body=json.dumps(data)) 
                output = response["Body"].read().decode()
                print(output)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
            else:
                st.session_state.option = "NLP"
                if rag:                    
                    
                    prompt_template = """Use the following pieces of Context to answer the Question at the end. If you don't know the answer, just say you don't know, don't try to make up an answer. Context:\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"""
                    PROMPT = PromptTemplate(
                          template=prompt_template, input_variables=["context", "question"]
                      )
                    chat_history = []
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm, 
                        retriever=retriever, 
                        # condense_question_prompt=standalone_question_prompt, 
                        return_source_documents=True, 
                        combine_docs_chain_kwargs={"prompt":PROMPT}
                    )
                    response = qa_chain({"question": user_input, "chat_history": chat_history})
                    output = response['answer']
                    source_docs = response['source_documents']
                    print(source_docs)
                    for doc in source_docs:
                        source.append(doc.metadata['source'])
                    
                else:
                    output = chatchain(user_input)["response"]
                print(output)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
        # when a file is uploaded we also send the content to the chatchain object and ask for confirmation
        elif uploaded_file is not None:
            if st.session_state.option == "Audio" and st.session_state.extract_audio:
                byteio = BytesIO(uploaded_file.getvalue())
                data = byteio.read()
                response = client.invoke_endpoint(EndpointName=whisper_endpoint_name, ContentType='audio/x-audio', Body=data)  
                output = json.loads(response['Body'].read())["text"]
                st.session_state['past'].append("I have uploaded an audio file. Plese extract the text from this audio file")
                st.session_state['generated'].append(output)
                content = "=== BEGIN AUDIO FILE ===\n"
                content += output
                content += "\n=== END AUDIO FILE ===\nPlease remember the audio file"
                output = chatchain(content)["response"]
                print(output)
                st.session_state.extract_audio = False
            elif st.session_state.option == "NLP":
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                content = "=== BEGIN FILE ===\n"
                content += stringio.read().strip()
                content += "\n=== END FILE ===\n\n Instruction: Please confirm that you have read that file by saying: 'Yes, I have read the file'"
                output = chatchain(content)["response"]
                st.session_state['past'].append("I have uploaded a file. Please confirm that you have read that file.")
                st.session_state['generated'].append(output)
        if len(source) != 0:
            df = pd.DataFrame(source, columns=['knowledge source'])
            st.data_editor(df)
            source = []    

        st.write(f"Currently using a {st.session_state.option} model")


    # this loop is responsible for displaying the chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                

with right_column:

    max_tokens= st.slider(
        min_value=8,
        max_value=1024,
        step=1,
        # value=200,
        label="Number of tokens to generate",
        key="max_token"
    )
    temperature = st.slider(
        min_value=0.1,
        max_value=2.5,
        step=0.1,
        # value=0.4,
        label="Temperature",
        key="temperature"
    )
    seed = st.slider(
        min_value=0,
        max_value=1000,
        # value=0,
        step=1,
        label="Random seed to use for the generation",
        key="seed"
    )

    
