import json

json_path = r"E:\archive\WLASL_v0.3.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON Structure:", list(data.keys()))  # Print the top-level keys
