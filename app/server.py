from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama  # Assuming llama_cpp is installed and used for the model
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

# Define the repository and filename
repo_id = "Nistep/bitnet_b1_58-3B-Q4_K_M-GGUF"
filename = "bitnet_b1_58-3b-q4_k_m.gguf"

# Download the file
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

llm = Llama(model_path=model_path)

# Initialize FastAPI app
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Use specific origins for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define a request body schema
class PromptRequest(BaseModel):
    prompt: str
# Define the endpoint
@app.post("/")
def generate_hello(request: PromptRequest):
    return "hi how are you "

# Define the endpoint
@app.post("/generate")
def generate_response(request: PromptRequest):
    try:
        # Get the prompt from the request
        prompt = request.prompt
        
        # Generate response
        response = llm(prompt)
        
        # Return the response text
        return {"response": response["choices"][0]["text"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
