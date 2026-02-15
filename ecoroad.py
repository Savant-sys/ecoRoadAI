from ultralytics import YOLO
import sys
import json
import shutil
from pathlib import Path


def summarize_detections(results):
    r = results[0]
    names = r.names
    counts = {}
    if r.boxes is not None and len(r.boxes) > 0:
        for cls_id in r.boxes.cls.tolist():
            name = names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1
    return counts

def eco_safety_rules(counts):
    cars = counts.get("car", 0) + counts.get("truck", 0) + counts.get("bus", 0) + counts.get("motorcycle", 0)
    people = counts.get("person", 0)
    stops = counts.get("stop sign", 0) + counts.get("traffic light", 0)

    tips = []
    risk = "low"

    # --- Simple eco scoring (demo-friendly) ---
    eco_score = 90  # start high, subtract penalties

    if cars >= 5:
        eco_score -= 25
        risk = "high"
        tips.append("Heavy traffic: smooth acceleration and longer following distance reduces braking and energy waste.")

    if stops >= 1:
        eco_score -= 15
        tips.append("Traffic control ahead: anticipate stops, coast early, and avoid rapid acceleration.")

    if people >= 1 and cars >= 1:
        eco_score -= 20
        risk = "high"
        tips.append("Pedestrians nearby: slow down early (safer and prevents inefficient stop-and-go).")

    if cars == 0 and people == 0:
        tips.append("Scene looks clear: maintain steady speed for better efficiency.")

    eco_score = max(0, min(100, eco_score))

    # --- CO2 impact label (rough demo metric) ---
    if eco_score >= 75:
        co2_label = "Low (efficient driving likely)"
    elif eco_score >= 50:
        co2_label = "Medium"
    else:
        co2_label = "High (stop-and-go likely)"

    if not tips:
        tips.append("General tip: gentle acceleration and steady cruising usually lowers emissions/energy use.")

    return risk, eco_score, co2_label, tips

def main():
    model = YOLO("yolov8n.pt")
    source = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    results = model.predict(source=source, save=True, imgsz=640, conf=0.25)

    # Find latest predict folder and copy annotated image to output/annotated.jpg
    runs_dir = Path("runs/detect")
    latest = sorted(runs_dir.glob("predict*"), key=lambda p: p.stat().st_mtime)[-1]
    annotated_src = latest / Path(source).name
    annotated_dst = Path("output/annotated.jpg")
    shutil.copyfile(annotated_src, annotated_dst)



    counts = summarize_detections(results)
    risk, eco_score, co2_label, tips = eco_safety_rules(counts)

    print("\n=== EcoRoad AI Summary ===")
    print("Detections:", counts if counts else "None")
    print("Risk Level:", risk)
    print("Eco Score:", eco_score)
    print("CO2 Impact Label:", co2_label)
    print("Eco + Safety Tips:")
    for t in tips:
        print("-", t)
    print("\nSaved annotated image to: runs/detect/predict/ (latest)")

    summary = {
        "source": source,
        "detections": counts,
        "risk_level": risk,
        "eco_score": eco_score,
        "co2_impact": co2_label,
        "tips": tips,
        "annotated_image": str(annotated_dst),
    }
    Path("output/summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved summary JSON to: output/summary.json")
    print("Saved annotated image to:", annotated_dst)

if __name__ == "__main__":
    main()
