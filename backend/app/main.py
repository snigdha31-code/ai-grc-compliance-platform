from fastapi import FastAPI
from app.routes import rag_routes
from app.models.db import engine
from app.models.audit_report import AuditReport
from app.models.db import Base
from app.routes import log_routes
from app.routes import grc_routes


Base.metadata.create_all(bind=engine) 
# Create tables if they don't exist
# Base is the root class for all our models
# We use SQLAlchemy ORM. Base.metadata.create_all() 
# automatically materializes the table schema into the database at application startup.

app = FastAPI(title="AI-Powered GRC Platform", version="0.1")
# FastAPI instance is created to define our web application. We set a title and version for API documentation purposes.
app.include_router(rag_routes.router, prefix="/rag", tags=["RAG"])
app.include_router(log_routes.router, prefix="/logs", tags=["Logs"])
app.include_router(grc_routes.router, prefix="/grc", tags=["GRC Automation"])

@app.get("/")
def root():
    return {
        "message": "AI-Powered GRC Platform is running.",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "ok"
    }
    
