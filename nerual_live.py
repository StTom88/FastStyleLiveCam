import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from models.transformer_net import TransformerNet
import os
import time

# Path to style models
MODEL_PATHS = {
    '1': (os.path.join("models", "mosaic.pth"), "Mosaic"),
    '2': (os.path.join("models", "candy.pth"), "Candy"),
    '3': (os.path.join("models", "rain_princess.pth"), "Rain Princess"),
    '4': (os.path.join("models", "starry-night.pth"), "Starry night"),
    '5': (os.path.join("models", "udnie.pth") , "Udnie"),
    '6': (os.path.join("models", "stormtrooper_26000.pth"), "Stormtrooper")
}

# Initialization
active_model = '1'
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print(f"Computation device: {device}")

# Output folder for saved snapshots
output_dir = "snapshots"
os.makedirs(output_dir, exist_ok=True)
existing_files = [f for f in os.listdir(output_dir) if f.startswith("snapshot_") and f.endswith(".jpg")]
existing_numbers = [int(f.split("_")[1]) for f in existing_files if f.split("_")[1].isdigit()]
snapshot_counter = max(existing_numbers, default=0) + 1


# Load style model
def load_model(model_path):
    style_model = TransformerNet()
    state_dict = torch.load(model_path, map_location="cpu")
    ignored = [k for k in state_dict.keys() if "running_mean" in k or "running_var" in k]
    for k in ignored:
        del state_dict[k]
    for k in list(state_dict.keys()):
        if k.startswith("saved_model."):
            state_dict[k.replace("saved_model.", "")] = state_dict.pop(k)
    style_model.load_state_dict(state_dict, strict=False)
    style_model.to(device)
    style_model.eval()
    return style_model

# Insteractions
print("\n Real-time video stylization using NN")
print("---------------------------Controls:----------------------")
print("1 = Mosaic")
print("2 = Candy")
print("3 = Rain Princess")
print("4 = Starry night")
print("5 = Udnie")
print("s = Snapshot")
print("q = Quit\n")

# Initial model load
model, model_name = load_model(MODEL_PATHS[active_model][0]), MODEL_PATHS[active_model][1]

# Image transform (convert to tensor and scale to [0, 255])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Main loop
while True:

    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # Convert from BGR to RGB
    img_tensor = transform(img).unsqueeze(0).to(device) # Convert to tensor

    with torch.no_grad():   
        output_tensor = model(img_tensor).cpu() 

        if model_name == "Stormtrooper":    # Temporary boost for undertrained model (Stormtrooper only)
            output_tensor = output_tensor * 200  # 200 * boost (output is dark without this)
            output_data = output_tensor.squeeze().clamp(0, 255).detach().numpy()
            output_data = output_data.transpose(1, 2, 0).astype('uint8')
        else:
            output_data = output_tensor.squeeze().clamp(0, 255).detach().numpy()
            output_data = output_data.transpose(1, 2, 0).astype('uint8')

    
    output_bgr = cv2.cvtColor(output_data, cv2.COLOR_RGB2BGR)

    # Prepare display frame
    clean_output = output_bgr.copy()
    original_resized = cv2.resize(frame, (output_bgr.shape[1], output_bgr.shape[0]))
    combined = cv2.hconcat([original_resized, output_bgr])

    # Info panel
    panel_height = 40
    panel = np.zeros((panel_height, combined.shape[1], 3), dtype=np.uint8)
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(panel, f"Styl: {model_name}   FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Show output
    output_with_panel = np.vstack([panel, combined])
    cv2.imshow("Real-time video stylization using NN", output_with_panel)

    # Key bindings
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = os.path.join(output_dir, f"snapshot_{snapshot_counter}_{model_name.lower().replace(' ', '_')}.jpg")
        cv2.imwrite(filename, clean_output)
        print(f"Image saved as: {filename}")
        snapshot_counter += 1
    elif chr(key) in MODEL_PATHS:
        active_model = chr(key)
        model, model_name = load_model(MODEL_PATHS[active_model][0]), MODEL_PATHS[active_model][1]
        print(f"\n Selected style: {model_name}")

# Release resources
cap.release()
cv2.destroyAllWindows()