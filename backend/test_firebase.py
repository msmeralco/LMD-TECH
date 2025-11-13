import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase
cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH"))
firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# Test write
doc_ref = db.collection("test").document("hello")
doc_ref.set({"message": "Firebase connected!", "timestamp": firestore.SERVER_TIMESTAMP})

# Test read
doc = doc_ref.get()
if doc.exists:
    print("✅ Firebase connected successfully!")
    print(f"Data: {doc.to_dict()}")
else:
    print("❌ Firebase connection failed")