from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from .prompt_generator import SUNOPromptGenerator
import uvicorn

app = FastAPI(title="SUNO提示词生成工具")
prompt_generator = SUNOPromptGenerator()

class PromptRequest(BaseModel):
    requirements: str

class FeedbackRequest(BaseModel):
    requirements: str
    generated_prompt: str
    feedback: str
    rating: int  # 1-5分

@app.post("/generate-prompt")
def generate_prompt(request: PromptRequest):
    try:
        prompt = prompt_generator.generate_prompt(request.requirements)
        return {"prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/provide-feedback")
def provide_feedback(request: FeedbackRequest):
    try:
        result = prompt_generator.train_with_feedback(
            request.requirements,
            request.generated_prompt,
            request.feedback,
            request.rating
        )
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "欢迎使用SUNO提示词生成工具，请访问/docs查看API文档"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)