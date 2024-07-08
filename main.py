from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Annotated
from typing import List
import os
#import asyncpg
import pandas as pd
from datetime import datetime
from database import SessionLocal, engine
from pydantic import BaseModel
import models
from fastapi.middleware.cors import CORSMiddleware
from notebook_executor import run_notebook
import requests


app = FastAPI()

models.Base.metadata.create_all(bind=engine)

class EnergyConsumption(BaseModel):
    
    id:int
    datetime:datetime
    building_id: int
    consumption:float
    
    
class Prediction(BaseModel):
    
    id:int
    datetime:datetime
    building_id: int
    predicted_value:float
    

# Konfigurisanje CORS-a
origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Zavistnost za sesiju baze podataka
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create directory for static files if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Helper function to connect to the database
#async def get_db_connection():
 #   conn = await asyncpg.connect(get_db())
  #  return conn

JUPYTER_TOKEN ='8b6adb97bb603c95516e48d8d812cee08302ea4878f23448'
import json
#vraca skriptu iz jupiter notebook
@app.get("/run-notebook")
def run_notebook_endpoint(notebook_url: str):
    try:
       # Prilagodite URL da ukaže na API i uključite token
        api_url = notebook_url.replace('/notebooks/', '/api/contents/')
        if '?' in api_url:
            api_url += f"&token={JUPYTER_TOKEN}"
        else:
            api_url += f"?token={JUPYTER_TOKEN}"
        
        # Pošaljite zahtev za dobijanje sadržaja beležnice
        response = requests.get(api_url)
        response.raise_for_status()
        notebook_content = response.json()
        if not notebook_content or 'content' not in notebook_content:
            raise HTTPException(status_code=500, detail="Nije moguće dobiti sadržaj beležnice.")
    
        # Dodavanje logovanja za proveru sadržaja
        #print("Sadržaj beležnice:", json.dumps(notebook_content, indent=2))
    #Xtest da posljem preko reacta
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Neuspelo preuzimanje beležnice: {e}")

    result = run_notebook(notebook_content['content'])
    data = json.loads(result)
    print(json.dumps(data))
# Pretvori JSON string u Python dictionary
    #result_data = json.loads(data['result'])
    # Ekstraktuj vrednost `text` polja unutar `outputs`
    texts = []
    for cell in data['cells']:
       if 'outputs' in cell:
          for output in cell['outputs']:
            if 'text' in output:
                texts.append(output['text'])

# Ispis rezultata
    for text in texts:
        print(text)
    return {"result": text}

class DateRange(BaseModel):
    start_date: datetime
    end_date: datetime
class NotebookRequest(BaseModel):
    notebook_url: str
    date_range: DateRange
class PredictionResult(BaseModel):
    time: datetime
    prediction: float
    
class UserCreate(BaseModel):
    username: str
    password: str

@app.post("/run_notebook")
def run_notebook_endpoint(request: NotebookRequest):
    notebook_url = request.notebook_url
    date_range = request.date_range
    try:
        # Prilagoditi URL da ukaže na API i uključite token
        api_url = notebook_url.replace('/notebooks/', '/api/contents/')
        if '?' in api_url:
            api_url += f"&token={JUPYTER_TOKEN}"
        else:
            api_url += f"?token={JUPYTER_TOKEN}"
        
        #Slanje zahteva za dobijanje sadržaja skripte
        response = requests.get(api_url)
        response.raise_for_status()
        notebook_content = response.json()
        print("Sadržaj skripte:", json.dumps(notebook_content, indent=2))
        if not notebook_content or 'content' not in notebook_content:
            raise HTTPException(status_code=500, detail="Nije moguće dobiti sadržaj skripte.")
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Neuspelo preuzimanje skripte: {e}")
    
    # Priprema parametara za Jupyter Notebook
    start_date = date_range.start_date
    end_date = date_range.end_date
    # Pretvaranje sadržaja skripte u JSON format i dodavanje parametara
    notebook_content_json = json.loads(json.dumps(notebook_content))
    parameters_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"start_date = '{start_date}'\n",
            f"end_date = '{end_date}'\n"
        ]
    }
    # Dodajte parametre kao prvu ćeliju u skripti
    notebook_content_json['content']['cells'].insert(0, parameters_cell)

    # Funkcija koja će izvršiti skriptu 
    result = run_notebook(notebook_content_json['content'])
    data = json.loads(result)

    #Ekstrakcija i vracanje tekstualnog rezultata
    texts = []
    for cell in data['cells']:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    text = output['text']
                # Uklanjanje zagrada oko teksta i dodavanje u listu
                    text_stripped = [line.strip("[]") for line in text]
                    texts.extend(text_stripped)
    for text in texts:
        print(text)
    return {"result": texts}

db_dependency = Annotated[Session, Depends(get_db)]

@app.post("/save_prediction")
def save_prediction(predictions: List[PredictionResult], db: db_dependency):
    print(predictions)
    try:
        for prediction in predictions:
            db_prediction = models.PredictionResult(
                time=prediction.time,
                prediction=prediction.prediction
            )
            db.add(db_prediction)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Greška prilikom upisa predikcije: {e}")
    return {"status": "Predikcije uspešno upisane"}

@app.get("/get_predictions")
def get_predictions(db: db_dependency):
    print( 'ddd')
    return db.query(models.PredictionResult).all()

@app.post("/login/")
def login(user: UserCreate, db:db_dependency):
    db_user = get_user_by_username(db, username=user.username)
    if db_user and db_user.password == user.password:
        return {"message": "Successfully logged in!"}
    else:
        raise HTTPException(status_code=400, detail="Invalid username or password")

@app.post("/register/")
def register(user: UserCreate, db:db_dependency):
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    return create_user(db=db, user=user)

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: UserCreate):
    db_user = models.User(username=user.username, password=user.password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user