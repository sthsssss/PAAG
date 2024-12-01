from ultralytics import SAM

# Load a model
model = SAM("weights/mobile_sam.pt")

# Display model information (optional)
# model.info()

# Run inference
model("doraemon.jpg")