from paligemma import PaliGemma
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Initialize the PaliGemma model
model = PaliGemma()

# Define the text prompt and image URL
text_prompt = "A beautiful animal."
image_url = "https://www.thesprucepets.com/thmb/wDwU14vPAAGa6sl9V0hdIrJggpI=/3600x0/filters:no_upscale():strip_icc()/cute-dog-breeds-we-can-t-get-enough-of-4589340-hero-04aba92c6fbb4651b7fa1f54823a1a6d.jpg"

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
