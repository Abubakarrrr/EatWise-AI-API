from fastapi import FastAPI, HTTPException,File,UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from pinecone import ServerlessSpec,Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
PINCENE_API_KEY = os.getenv("PINECONE_API_KEY")
# Initialize FastAPI app
app = FastAPI()

# Initialize SentenceTransformer model for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone connection
pc =Pinecone(api_key=PINCENE_API_KEY)

# Check if the 'eatwise' index exists, otherwise create it
if 'eatwise' not in pc.list_indexes().names():
    pc.create_index(
        name='eatwise',
        dimension=384,  # Embedding dimension for 'all-MiniLM-L6-v2'
        metric='euclidean',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Define Pydantic model for Meal data
class Meal(BaseModel):
    title: str
    description: str
    ingredients: list[str]
    steps: list[str]
    category: str
    calories: int
    protein: int
    carbs: int
    fats: int


def extract_meal_data_from_pdf(pdf_file):
    try:
        # Initialize PDF reader
        reader = PdfReader(pdf_file)
        raw_text = ""

        # Extract text from all pages
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"

        # Process and split meals correctly based on special symbol
        meal_chunks = process_meal_text(raw_text)

        return meal_chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

# Helper function to process meal text
def process_meal_text(text: str):
    try:
        # Clean the text to remove unnecessary line breaks and extra spaces
        text = text.replace("\n", " ").strip()

        # Split meals by the special symbol "@@END@@"
        meals = text.split("@@END@@")

        # Clean up the chunks to remove extra spaces
        meals = [meal.strip() for meal in meals if meal.strip()]  # Remove empty or invalid chunks

        return meals
    except Exception as e:
        print(f"Error processing meal text: {str(e)}")
        return None


# API endpoint to upload PDF and store meal embeddings
@app.post("/upload-pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    try:
        # Extract meals from the uploaded PDF
        meals = extract_meal_data_from_pdf(pdf_file.file)
        # print(meals)
    
        # Loop through the meals and create embeddings for each meal
        for i, meal_text in enumerate(meals):
            # Generate embedding for the entire meal text (single chunk)
            embedding = model.encode([meal_text])[0]  # Encoding the entire meal as one chunk
            
            # Upsert the embedding into Pinecone (store meal text as metadata)
            upsert_data = [
                (f"meal-{i}", embedding.tolist(), {"meal_text": meal_text})
            ]
            pc.Index('eatwise').upsert(vectors=upsert_data)
        
        return {"message": "Meal data from PDF has been successfully processed and stored in Pinecone."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
