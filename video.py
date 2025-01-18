import os
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def draw_bounding_box(image, coordinates, label, width, height):
    global label_colors
    y1, x1, y2, x2 = coordinates
    y1, x1, y2, x2 = map(round, (y1*height, x1*width, y2*height, x2*width))

    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    text_width, text_height = text_size

    text_x = x1 + 2
    text_y = y1 - 5

    font_scale = 1
    label_rect_width = text_width + 8
    label_rect_height = int(text_height * font_scale)

    color = label_colors.get(label, None)
    if color is None:
        color = np.random.randint(0, 256, (3,)).tolist()
        label_colors[label] = color

    cv2.rectangle(image, (x1, y1 - label_rect_height), (x1 + label_rect_width, y1), color, -1)

    thickness = 2
    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = PaliGemmaProcessor.from_pretrained(model_id)

input_video = 'input_video.mp4'
prompt = "detect person, bag, black trouser"
output_file = 'output_video.avi'

# Open the input video file.
cap = cv2.VideoCapture(input_video)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

label_colors = {}

while(True):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL Image.
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Send text prompt and image as input.
    inputs = processor(text=prompt, images=img,
                      padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
    model.to(device)
    inputs = inputs.to(dtype=model.dtype)

    # Get output.
    with torch.no_grad():
        output = model.generate(**inputs, max_length=496)

    paligemma_response = processor.decode(output[0], skip_special_tokens=True)[len(prompt):].lstrip("\n")
    print(paligemma_response)
    detections = paligemma_response.split(" ; ")

    # Parse the output bounding box coordinates
    parsed_coordinates = []
    labels = []

    for item in detections:
        detection = item.replace("<loc", "").split()

        if len(detection) >= 2:
          coordinates_str = detection[0].replace(",", "")
          label = detection[1]
          if "<seg" in label:
            continue
          else:
            labels.append(label)
        else:
          # No label detected, skip the iteration.
          continue

        coordinates = coordinates_str.split(">")
        coordinates = coordinates[:4]

        if coordinates[-1] == '':
            coordinates = coordinates[:-1]


        coordinates = [int(coord)/1024 for coord in coordinates]
        parsed_coordinates.append(coordinates)

    width = img.size[0]
    height = img.size[1]

    # Draw bounding boxes on the frame
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for coordinates, label in zip(parsed_coordinates, labels):
      output_frame = draw_bounding_box(frame, coordinates, label, width, height)

    # Write the frame to the output video
    out.write(output_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture, output video writer, and destroy the window
cap.release()
out.release()
cv2.destroyAllWindows()
print("Output video " + output_file + " saved to disk.")