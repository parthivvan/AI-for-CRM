import json
import os
import time
import random
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

def preprocess(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
    arr = arr.transpose(2, 0, 1)  # CHW
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return arr[np.newaxis, ...]  # (1, 3, 224, 224)

def evaluate_split(session, labels, split_dir: Path):
    if not split_dir.exists():
        return None

    y_true = []
    y_pred = []
    
    start_time = time.time()
    count = 0
    
    # Traverse directories representing classes
    for class_name in labels:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        
        all_imgs = [p for p in class_dir.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Sample 50 images per class for train split to speed up execution
        if "train" in split_dir.name.lower():
            random.seed(42)
            all_imgs = random.sample(all_imgs, min(len(all_imgs), 50))
            
        for img_path in all_imgs:
            try:
                tensor = preprocess(img_path)
                input_name = session.get_inputs()[0].name
                raw_output = session.run(None, {input_name: tensor})
                scores = np.array(raw_output[0][0], dtype=np.float32)
                
                # Apply softmax
                probs = np.exp(scores - np.max(scores))
                probs /= probs.sum()
                
                pred_idx = np.argmax(probs)
                pred_label = labels[pred_idx]
                
                y_true.append(class_name)
                y_pred.append(pred_label)
                count += 1
            except Exception as e:
                # Silently skip corrupted images
                pass
                
    elapsed = time.time() - start_time
    
    # Calculate metrics per class
    metrics = {}
    total_correct = 0
    
    for class_name in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == class_name and p == class_name)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != class_name and p == class_name)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == class_name and p != class_name)
        support = sum(1 for t in y_true if t == class_name)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "correct": tp
        }
        total_correct += tp
        
    accuracy = total_correct / count if count > 0 else 0.0
    
    return {
        "accuracy": round(accuracy, 4),
        "total_images": count,
        "elapsed_seconds": round(elapsed, 2),
        "ms_per_image": round((elapsed / count) * 1000, 2) if count > 0 else 0.0,
        "metrics_per_class": metrics
    }

def main():
    model_path = Path("models/current/mobilenetv3_large_skin_ai.onnx")
    labels_path = Path("models/current/labels.json")
    dataset_dir = Path("ALL DATASETS FOR SKIN INCLUDING FINAL/Final_Dataset")
    
    print(f"Loading ONNX session: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    print(f"Model Labels: {labels}")
    
    results = {}
    for split in ["test", "validation", "train"]:
        print(f"Evaluating {split} split...")
        res = evaluate_split(session, labels, dataset_dir / split)
        if res:
            results[split] = res
            print(f"{split.capitalize()} split accuracy: {res['accuracy']:.4f} ({res['total_images']} images)")
            
    out_report = Path("scratch/onnx_evaluation_report.json")
    out_report.write_text(json.dumps(results, indent=2))
    print(f"Report written to {out_report}")

if __name__ == "__main__":
    main()
