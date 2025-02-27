from fastapi import FastAPI, HTTPException,File,UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import pinecone 
from pinecone import ServerlessSpec,Pinecone

# Initialize FastAPI app
app = FastAPI()

# Initialize SentenceTransformer model for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone connection
pc =Pinecone(api_key="pcsk_3uYCXJ_HPbLLtfT1AEQcJBQnnKUvtQfCfWT6igB7cmDLdvS2dUtf7gt5goUFsydoPyjffk")

# Check if the 'eatwise' index exists, otherwise create it
if 'eatwise' not in pc.list_indexes().names():
    pc.create_index(
        name='eatwise',
        dimension=1536,  # Embedding dimension for 'all-MiniLM-L6-v2'
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

# Helper function to parse PDF file and extract meal data
def extract_meal_data_from_pdf(pdf_file):
    try:
        # Initialize PDF reader
        reader = PdfReader(pdf_file)
        meals = []
        
        # Loop through all pages of the PDF
        for page in reader.pages:
            text = page.extract_text()
            
            # Process the text to extract meal data (this part will depend on your PDF structure)
            meal_data = process_meal_text(text)
            if meal_data:
                meals.append(meal_data)
                
        return meals
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

# Helper function to process raw PDF text into structured meal data
def process_meal_text(text: str):
    try:
        print(text)
        # Split meals by blank lines (assuming each meal is separated by a blank line)
        meals_raw = [meal.strip() for meal in text.split("\n\n") if meal.strip()]
        # print(meals_raw)
        meals = []
        for meal_text in meals_raw: 
            # Treat each meal as a single chunk of text
            meals.append(meal_text)
        
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
        # Loop through the meals and create embeddings for each meal
        # for i, meal_text in enumerate(meals):
        #     # Generate embedding for the entire meal text (single chunk)
        #     embedding = model.encode([meal_text])[0]  # Encoding the entire meal as one chunk
            
        #     # Upsert the embedding into Pinecone (store meal text as metadata)
        #     upsert_data = [
        #         (f"meal-{i}", embedding.tolist(), {"meal_text": meal_text})
        #     ]
        #     pinecone.index('eatwise').upsert(vectors=upsert_data)
        
        # return {"message": "Meal data from PDF has been successfully processed and stored in Pinecone."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
