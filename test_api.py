import requests
import sys

def test():
    url = "http://localhost:5000/api/analyze"
    image_path = r"d:\Projects\deceptiui\dataset\processed\images\10011.jpg" # Using one image as test
    
    with open(image_path, "rb") as img:
        files = {"image": img}
        print(f"Sending POST to {url} with {image_path}")
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Success!")
        print("Predictions:", data.get("predictions"))
        print("Top Prediction:", data.get("top_prediction"))
        patterns = data.get("patterns", {})
        print(f"Total Patterns with generated evidence: {len(patterns)}")
        for label, pat in patterns.items():
            print(f"--- Pattern: {label} ({pat['confidence']*100:.1f}%) ---")
            print(f"    Explanation: {pat['explanation'][:80]}...")
            print(f"    Text Evidence count: {len(pat.get('text_evidence', []))}")
            print(f"    Has Visual Evidence: {'visual_evidence' in pat and len(pat['visual_evidence']) > 50}")
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    test()
