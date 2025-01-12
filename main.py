from paligemma import PaliGemma
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Initialize the PaliGemma model
model = PaliGemma()

# Define the text prompt and image URL
text_prompt = "<image>caption en"
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"

# Run the PaliGemma model
output = model.run(text_prompt, image_url)
# Print the generated output
print(output)

@app.post("/generate")
async def generate(task: str, image: UploadFile = File(...)):
    contents = await image.read()
    with open("input.jpg", "wb") as f:
        f.write(contents)
    output = model.run(task, "input.jpg")
    return {"output": output}

@app.post("/generate2")
async def generate(task: str, image_url: str):
    output = model.run(task, image_url)
    return {"output": output}