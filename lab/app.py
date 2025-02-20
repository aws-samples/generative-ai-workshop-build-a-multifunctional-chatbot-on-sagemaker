import streamlit as st
import boto3
import json
import os
import base64
from PIL import Image
import io
from datetime import datetime

# Default compression settings
DEFAULT_MAX_SIZE_KB = 2048
DEFAULT_MAX_DIMENSION = 800

class ImageProcessor:
    """Handle image processing operations"""
    @staticmethod
    def compress_image(image_bytes, max_size_kb=DEFAULT_MAX_SIZE_KB, max_dimension=DEFAULT_MAX_DIMENSION):
        """Compress image to specified size"""
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Resize image
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        # Compress with quality adjustment
        output = io.BytesIO()
        quality = 95
        while quality > 5:
            output.seek(0)
            output.truncate()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            if len(output.getvalue()) <= max_size_kb * 1024:
                break
            quality -= 5
        
        return output.getvalue()

class FileProcessor:
    """Handle different types of file processing"""
    
    SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg']
    SUPPORTED_TEXT_TYPES = ['txt', 'pdf', 'docx', 'json']
    
    @staticmethod
    def process_file(file, file_type):
        """Process different types of files"""
        if file_type in FileProcessor.SUPPORTED_IMAGE_TYPES:
            return FileProcessor.process_image(file)
        elif file_type in FileProcessor.SUPPORTED_TEXT_TYPES:
            return FileProcessor.process_text(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    @staticmethod
    def process_image(file):
        """Process image files"""
        return {
            'type': 'image',
            'content': ImageProcessor.compress_image(file.getvalue()),
            'original_name': file.name
        }
    
    @staticmethod
    def process_text(file):
        """Process text files"""
        if file.name.endswith('.pdf'):
            # Add PDF processing logic here
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
        elif file.name.endswith('.docx'):
            # Add DOCX processing logic here
            from docx import Document
            document = Document(file)
            text = "\n".join([paragraph.text for paragraph in document.paragraphs])
            
        elif file.name.endswith('.json'):
            # Add JSON processing logic here
            text = json.load(file)
            text = json.dumps(text, indent=2)
            
        else:  # txt files
            text = file.getvalue().decode('utf-8')
            
        return {
            'type': 'text',
            'content': text,
            'original_name': file.name
        }
        
class ConfigManager:
    """Manage application configuration"""
    @staticmethod
    def load_config():
        """Load configuration from file"""
        try:
            with open('demo-dev/utils/tmp_config.json', 'r') as f:
                config = json.load(f)
                return (
                    config['kb_id'],
                    config['nova_pro_profile_arn'],
                    config['nova_pro_model_id'],
                    config['region_name'],
                    config['sagemaker_endpoint'],
                    config['sagemaker_ep_arn']
                )
        except Exception as e:
            st.error(f"Configuration error: {str(e)}")
            st.stop()

class ModelManager:
    """Handle model operations and configurations"""
    def __init__(self):
        # Initialize AWS clients
        self.bedrock_agent = boto3.client('bedrock-agent-runtime')
        self.bedrock_runtime = boto3.client('bedrock-runtime')
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')
        
        # Load configuration
        self.KB_ID, self.NOVA_PRO_PROFILE_ARN, self.NOVA_PRO_MODEL_ID, \
        self.REGION_NAME, self.SAGEMAKER_ENDPOINT, self.SAGEMAKER_EP_ARN = ConfigManager.load_config()

        # Define model configurations
        self.MODELS = {
            "Amazon Titan Nova Pro": {
                "model_arn": self.NOVA_PRO_PROFILE_ARN,
                "model_id": self.NOVA_PRO_MODEL_ID,
                "type": "nova",
                "supports_image": True
            },
            "Claude 3.5 Sonnet": {
                "model_arn": "arn:aws:bedrock:us-west-2:010117700078:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "type": "claude",
                "supports_image": True
            },
            "DeepSeek R1 Distill Qwen 1.5B": {
                "model_arn": self.SAGEMAKER_EP_ARN,
                "endpoint_name": self.SAGEMAKER_ENDPOINT,
                "type": "deepseek",
                "supports_image": False,
                "supports_kb": True
            }
        }

    def generate_request(self, model_type, query, image_data=None, model_params=None):
        """Generate model-specific request body"""
        if model_type == "claude":
            content = [{"type": "text", "text": query}]
            if image_data:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                })
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": model_params.get("max_new_tokens", 1000),
                "temperature": model_params.get("temperature", 0.7),
                "messages": [{"role": "user", "content": content}]
            }
        elif model_type == "nova":
            content = []
            if image_data:
                content.append({
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": image_data}
                    }
                })
            content.append({"text": query})
            return {
                "schemaVersion": "messages-v1",
                "messages": [{"role": "user", "content": content}],
                "inferenceConfig": {
                    "max_new_tokens": model_params.get("max_new_tokens", 1000),
                    "temperature": model_params.get("temperature", 0.7),
                    "top_k": model_params.get("top_k", 50),
                    "top_p": model_params.get("top_p", 0.9)
                }
            }
        else:  # deepseek
            return {
                "inputs": query,
                "parameters": {
                    "max_new_tokens": model_params.get("max_new_tokens", 1000),
                    "temperature": model_params.get("temperature", 0.7),
                    "top_k": model_params.get("top_k", 50),
                    "top_p": model_params.get("top_p", 0.9)
                }
            }

    def process_query(self, model_config, query, processed_files=None, use_kb=False, model_params=None):
        """Process query with selected model"""
        try:
            # For Deepseek model, check if there are any image files
            if model_config["type"] == "deepseek" and processed_files:
                has_image = any(file['type'] == 'image' for file in processed_files)
                if has_image:
                    return ("The DeepSeek model only supports text processing. "
                           "For queries involving images, please use either Claude or Nova Pro. "
                           "If you wish to use DeepSeek, please upload text files only.", None)
    
            # Handle multiple files
            if processed_files:
                # For Claude model
                if model_config["type"] == "claude":
                    content = [{"type": "text", "text": query}]
                    text_contents = []
                    
                    for file in processed_files:
                        if file['type'] == 'image':
                            if not model_config["supports_image"]:
                                continue
                            # Convert bytes to base64 string
                            image_base64 = base64.b64encode(file['content']).decode('utf-8')
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            })
                        else:  # text files
                            text_contents.append(f"\n=== Content from {file['original_name']} ===\n")
                            content_preview = file['content'][:1000] + ("..." if len(file['content']) > 1000 else "")
                            text_contents.append(content_preview)
                    
                    if text_contents:
                        prompt = f"""Please analyze the following file content and answer the question:
    
    File Content:
    {' '.join(text_contents)}
    
    Question: {query}
    
    Please provide a clear and concise answer focusing on the main points of the content."""
                        content[0]["text"] = prompt
                    
                    request_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": model_params.get("max_new_tokens", 1000),
                        "temperature": model_params.get("temperature", 0.7),
                        "messages": [{"role": "user", "content": content}]
                    }
                    
                # For Nova model
                elif model_config["type"] == "nova":
                    content = []
                    text_contents = []
                    
                    for file in processed_files:
                        if file['type'] == 'image':
                            if not model_config["supports_image"]:
                                continue
                            # Convert bytes to base64 string for Nova
                            image_bytes_base64 = base64.b64encode(file['content']).decode('utf-8')
                            content.append({
                                "image": {
                                    "format": "jpeg",
                                    "source": {"bytes": image_bytes_base64}
                                }
                            })
                        else:  # text files
                            text_contents.append(f"\n=== Content from {file['original_name']} ===\n")
                            content_preview = file['content'][:1000] + ("..." if len(file['content']) > 1000 else "")
                            text_contents.append(content_preview)
                    
                    if text_contents:
                        prompt = f"""Please analyze the following file content and answer the question:
    
    File Content:
    {' '.join(text_contents)}
    
    Question: {query}
    
    Please provide a clear and concise answer focusing on the main points of the content."""
                    else:
                        prompt = query
                        
                    content.append({"text": prompt})
                    request_body = {
                        "schemaVersion": "messages-v1",
                        "messages": [{"role": "user", "content": content}],
                        "inferenceConfig": {
                            "max_new_tokens": model_params.get("max_new_tokens", 1000),
                            "temperature": model_params.get("temperature", 0.7),
                            "top_k": model_params.get("top_k", 50),
                            "top_p": model_params.get("top_p", 0.9)
                        }
                    }
                    
                # For Deepseek model
                else:
                    # Combine all text content with the query
                    text_contents = []
                    for file in processed_files:
                        if file['type'] == 'text':
                            text_contents.append(f"\n=== Content from {file['original_name']} ===\n")
                            content_preview = file['content'][:1000] + ("..." if len(file['content']) > 1000 else "")
                            text_contents.append(content_preview)
                    
                    if text_contents:
                        prompt = f"""Please analyze the following file content and answer the question:
    
    File Content:
    {' '.join(text_contents)}
    
    Question: {query}
    
    Please provide a clear and concise answer focusing on the main points of the content."""
                    else:
                        prompt = query
                    
                    request_body = {
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": model_params.get("max_new_tokens", 1000),
                            "temperature": model_params.get("temperature", 0.7),
                            "top_k": model_params.get("top_k", 50),
                            "top_p": model_params.get("top_p", 0.9)
                        }
                    }
                
                # Special handling for Deepseek with KB
                if model_config["type"] == "deepseek" and use_kb:
                    try:
                        kb_response = self.bedrock_agent.retrieve(
                            knowledgeBaseId=self.KB_ID,
                            retrievalQuery={"text": query},
                            retrievalConfiguration={
                                "vectorSearchConfiguration": {
                                    "numberOfResults": 3
                                }
                            }
                        )
                        
                        contexts = []
                        citations = []
                        
                        if 'retrievalResults' in kb_response:
                            for result in kb_response['retrievalResults']:
                                if 'content' in result:
                                    content = result['content']
                                    if isinstance(content, dict):
                                        text = content.get('text', '')
                                    else:
                                        text = str(content)
                                    contexts.append(text)
                                    citations.append({
                                        'retrievedReferences': [{'content': text}]
                                    })
        
                        if not contexts:
                            return "No relevant content found in knowledge base.", None
        
                        prompt = f"""Review the following AWS documentation and answer the user's question.
                                    Make sure your answer is:
                                    1. Accurate according to the provided documentation
                                    2. Directly addresses the user's question
                                    3. Clear and easy to understand
                                    
                                    AWS Documentation:
                                    {' '.join(contexts)}
                                    
                                    User Question:
                                    {query}"""
                                    
                        request_body = self.generate_request(
                            model_config["type"], 
                            prompt,
                            None,
                            model_params
                        )
                        
                        generated_text, _ = self.process_deepseek_query(model_config, request_body)
                        return generated_text, citations
                        
                    except Exception as e:
                        print(f"KB retrieval error: {str(e)}")
                        raise
                
                # Process the request based on model type
                if model_config["type"] == "deepseek":
                    return self.process_deepseek_query(model_config, request_body)
                else:
                    return self.process_bedrock_query(model_config, request_body)
                    
            # If no files, process as normal query
            elif use_kb and model_config.get("supports_kb", True):
                return self.process_kb_query(model_config, query)
            else:
                request_body = self.generate_request(
                    model_config["type"], 
                    query,
                    None,
                    model_params
                )
                
                if model_config["type"] == "deepseek":
                    return self.process_deepseek_query(model_config, request_body)
                else:
                    return self.process_bedrock_query(model_config, request_body)
                    
        except Exception as e:
            raise Exception(f"Query processing error: {str(e)}")


    def process_kb_query(self, model_config, query):
        """Process knowledge base query"""
        if model_config["type"] == "deepseek":
            # For Deepseek model, first retrieve from KB then process separately
            try:
                kb_response = self.bedrock_agent.retrieve(
                    knowledgeBaseId=self.KB_ID,
                    retrievalQuery={"text": query},
                    retrievalConfiguration={
                        "vectorSearchConfiguration": {
                            "numberOfResults": 3
                        }
                    }
                )
                
                contexts = []
                citations = []
                
                if 'retrievalResults' in kb_response:
                    for result in kb_response['retrievalResults']:
                        if 'content' in result:
                            content = result['content']
                            if isinstance(content, dict):
                                text = content.get('text', '')
                            else:
                                text = str(content)
                            contexts.append(text)
                            citations.append({
                                'retrievedReferences': [{'content': text}]
                            })
    
                if not contexts:
                    return "No relevant content found in knowledge base.", None
    
                prompt = f"""Review the following AWS documentation and answer the user's question.
                            Make sure your answer is:
                            1. Accurate according to the provided documentation
                            2. Directly addresses the user's question
                            3. Clear and easy to understand
                            
                            AWS Documentation:
                            {' '.join(contexts)}
                            
                            User Question:
                            {query}"""
                
                request_body = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 1000,
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.9
                    }
                }
                
                generated_text, _ = self.process_deepseek_query(model_config, request_body)
                return generated_text, citations
                
            except Exception as e:
                print(f"KB retrieval error: {str(e)}")
                raise
                
        else:
            # For other models, use the original retrieve_and_generate
            response = self.bedrock_agent.retrieve_and_generate(
                input={"text": query},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.KB_ID,
                        "modelArn": model_config["model_arn"],
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": 3
                            }
                        }
                    }
                }
            )
            return response['output'], response.get('citations')

    def process_deepseek_query(self, model_config, request_body):
        """Process Deepseek model query"""
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=model_config["endpoint_name"],
            ContentType='application/json',
            Body=json.dumps(request_body)
        )
        response_text = response['Body'].read().decode('utf-8')
        response_body = json.loads(response_text)
        
        if isinstance(response_body, list):
            return response_body[0]['generated_text'], None
        return response_body.get('generated_text', str(response_body)), None

    def process_bedrock_query(self, model_config, request_body):
        """Process Bedrock model query"""
        response = self.bedrock_runtime.invoke_model(
            modelId=model_config["model_arn"],
            body=json.dumps(request_body)
        )
        response_body = json.loads(response['body'].read())
        
        if model_config["type"] == "claude":
            return response_body['content'][0]['text'], None
        return response_body["output"]["message"]["content"][0]["text"], None

class UI:
    """Handle UI components and interactions"""
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.setup_page()

    def setup_page(self):
        """Setup main page layout"""
        st.set_page_config(page_title="MultiFunctional Chatbot with Bedrock KB and Deepseek Models", layout="wide")
        st.title("MultiFunctional Chatbot with Bedrock KB and Deepseek Models")
        
        # Main content
        self.selected_model = st.selectbox(
            "Select Generation Model",
            options=list(self.model_manager.MODELS.keys()),
            index=0
        )
        self.MODEL_CONFIG = self.model_manager.MODELS[self.selected_model]
        
        # Sidebar settings
        with st.sidebar:
            self.setup_sidebar()

        self.setup_main_content()

    def setup_sidebar(self):
        """Setup sidebar components"""
        st.markdown("### Image Compression Settings")
        self.max_size_kb = st.number_input(
            "Max Image Size (KB)", 
            min_value=30, 
            max_value=2048, 
            value=DEFAULT_MAX_SIZE_KB
        )
        self.max_dimension = st.number_input(
            "Max Image Dimension", 
            min_value=200, 
            max_value=1600, 
            value=DEFAULT_MAX_DIMENSION
        )
    
        # Add model parameter controls with both slider and number input
        st.markdown("### Model Parameters")
        self.model_params = {}
        
        # Max New Tokens
        self.model_params["max_new_tokens"] = st.slider(
            "Max New Tokens ",
            min_value=1,
            max_value=4096,
            value=1000
        )
        self.model_params["max_new_tokens"] = st.number_input(
            "Max New Tokens  ",
            min_value=1,
            max_value=4096,
            value=self.model_params["max_new_tokens"]
        )
        
        # Temperature
        self.model_params["temperature"] = st.slider(
            "Temperature ",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        self.model_params["temperature"] = st.number_input(
            "Temperature  ",
            min_value=0.0,
            max_value=1.0,
            value=self.model_params["temperature"],
            format="%.1f"
        )
        
        # Top K
        self.model_params["top_k"] = st.slider(
            "Top K ",
            min_value=1,
            max_value=500,
            value=50
        )
        self.model_params["top_k"] = st.number_input(
            "Top K  ",
            min_value=1,
            max_value=500,
            value=self.model_params["top_k"]
        )
        
        # Top P
        self.model_params["top_p"] = st.slider(
            "Top P ",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1
        )
        self.model_params["top_p"] = st.number_input(
            "Top P  ",
            min_value=0.0,
            max_value=1.0,
            value=self.model_params["top_p"],
            format="%.1f"
        )

    def setup_main_content(self):
        """Setup main content area"""
        st.markdown("---")
        
        # File upload section
        st.subheader("File Upload")
        uploaded_files = st.file_uploader(
            "Upload files (images and/or documents)",
            type=FileProcessor.SUPPORTED_IMAGE_TYPES + FileProcessor.SUPPORTED_TEXT_TYPES,
            accept_multiple_files=True
        )
        
        self.processed_files = []
        if uploaded_files:
            for file in uploaded_files:
                try:
                    file_type = file.name.split('.')[-1].lower()
                    processed_file = FileProcessor.process_file(file, file_type)
                    self.processed_files.append(processed_file)
                    
                    # Display preview based on file type
                    if processed_file['type'] == 'image':
                        st.image(file, caption=f"Original Image: {file.name}", use_container_width=True)
                        compressed_size_kb = len(processed_file['content'])/1024
                        st.info(f"Image compressed to {compressed_size_kb:.1f} KB")
                    else:
                        with st.expander(f"Preview: {file.name}"):
                            st.text(processed_file['content'][:1000] + "..." if len(processed_file['content']) > 1000 else processed_file['content'])
                            
                except Exception as e:
                    st.error(f"Error processing file {file.name}: {str(e)}")
        
        self.query = st.text_area(
            "Enter your query:",
            height=150,
            placeholder="Enter your question here..."
        )
        
        st.markdown("---")
        
        self.use_kb = st.checkbox("Use Knowledge Base", value=True) if not self.processed_files else False
        
        # Create placeholders for button and status
        button_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if button_placeholder.button("Search", type="primary", use_container_width=True, key="search_button"):
            # Clear the button and show status
            button_placeholder.empty()
            status_placeholder.info("Searching...")
            
            try:
                self.handle_search()
            finally:
                # Clear the status message
                status_placeholder.empty()
                # Show the button again
                button_placeholder.button(
                    "Search",
                    type="primary",
                    use_container_width=True,
                    key="search_button_after"
                )

    def handle_image_upload(self):
        """Handle image upload and processing"""
        try:
            if not self.MODEL_CONFIG["supports_image"]:
                st.error(f"Selected model ({self.selected_model}) does not support image analysis")
                st.stop()
                
            st.image(self.uploaded_file, caption="Original Image", use_container_width=True)
            self.image_bytes = ImageProcessor.compress_image(
                self.uploaded_file.getvalue(),
                max_size_kb=self.max_size_kb,
                max_dimension=self.max_dimension
            )
            
            compressed_size_kb = len(self.image_bytes)/1024
            if compressed_size_kb > self.max_size_kb:
                st.error(f"Image is still too large ({compressed_size_kb:.1f}KB)")
                st.stop()
            
            self.image_base64 = base64.b64encode(self.image_bytes).decode('utf-8')
            st.info(f"Image compressed to {compressed_size_kb:.1f} KB")
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.stop()

    def handle_search(self):
        """Handle search button click"""
        if not self.query.strip():
            st.warning("Please enter a query")
            return
    
        try:
            generated_text, citations = self.model_manager.process_query(
                self.MODEL_CONFIG,
                self.query,
                getattr(self, 'processed_files', None),
                self.use_kb if not getattr(self, 'processed_files', None) else False,
                self.model_params
            )
    
            st.subheader("Generated Answer:")
            st.write(generated_text)
            self.display_citations(citations)
    
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Full error details:")
            st.write(e)

    @staticmethod
    def display_citations(citations):
        """Display citation information"""
        if citations:
            st.subheader("Retrieved References:")
            for i, citation in enumerate(citations, 1):
                with st.expander(f"Reference {i}", expanded=False):
                    if 'retrievedReferences' in citation:
                        for ref in citation['retrievedReferences']:
                            st.write(ref['content'])
                            st.write("---")

def main():
    """Main application entry point"""
    model_manager = ModelManager()
    ui = UI(model_manager)

if __name__ == "__main__":
    main()