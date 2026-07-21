import os
import random
import httpx
import asyncio
import base64
import mimetypes

API_URL = "http://127.0.0.1:8000/analyze"
API_KEY = "dev-ai-service-key"
DATASET_DIR = "ALL DATASETS FOR SKIN INCLUDING FINAL/Final_Dataset/validation"

async def test_images():
    classes = ["acne", "dryness", "hair_thinning", "pigmentation", "redness"]
    
    print(f"{'True Class':<15} | {'Predicted Class':<15} | {'Conf':<6} | {'Match'}")
    print("-" * 60)
    
    total = 0
    correct = 0
    
    async with httpx.AsyncClient(timeout=30) as client:
        for cls in classes:
            class_dir = os.path.join(DATASET_DIR, cls)
            if not os.path.exists(class_dir):
                continue
                
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            # Pick 3 random images
            random.seed(42) # Deterministic for reproducibility
            selected = random.sample(images, min(3, len(images)))
            
            for img_name in selected:
                img_path = os.path.join(class_dir, img_name)
                
                # Create data URI
                mime_type = mimetypes.guess_type(img_path)[0] or "image/jpeg"
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("ascii")
                data_uri = f"data:{mime_type};base64,{encoded}"
                
                payload = {
                    "consultation_id": f"test_{total}",
                    "branch_id": "pilot-branch-001",
                    "client_id": "client_1",
                    "image_url": data_uri,
                    "image_type": "hair" if cls == "hair_thinning" else "skin"
                }
                
                response = await client.post(
                    API_URL,
                    headers={"X-API-Key": API_KEY},
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    pred = data["detected_flags"][0] if data["detected_flags"] else "None"
                    conf = data["confidence"]
                    match = "YES" if pred == cls else "NO"
                    if match == "YES":
                        correct += 1
                    total += 1
                    print(f"{cls:<15} | {pred:<15} | {conf:.2f} | {match}")
                else:
                    print(f"Error for {img_name}: {response.status_code} - {response.text}")
                    
    print("-" * 60)
    print(f"Accuracy on small sample: {correct}/{total} ({(correct/total)*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(test_images())
