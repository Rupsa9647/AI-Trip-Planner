import os
from dotenv import load_dotenv
from typing import Any, Optional
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq


class ConfigLoader:
    def __init__(self):
        print("Loaded config.....")
        self.config = load_config()
    
    def __getitem__(self, key):
        return self.config[key]


class ModelLoader(BaseModel):
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        self.config = ConfigLoader()
    
    class Config:
        arbitrary_types_allowed = True
    
    def load_llm(self):
        """
        Load and return the Groq LLM model.
        """
        print("LLM loading...")
        print("Loading LLM from Groq..............")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        model_name = self.config["llm"]["groq"]["model_name"]
        llm = ChatGroq(model=model_name, api_key=groq_api_key)
        
        return llm
