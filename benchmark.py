import os
import sys
import requests
import time
import json
import glob

# Configuration
ENDPOINTS = [
    # {
    #     "name": "python",
    #     "url": "http://localhost:5000/predict",
    #     "log_dir": "benchmark_logs/python"
    # },
    {
        "name": "rust",
        "url": "http://localhost:9050/predict",
        "log_dir": "benchmark_logs/rust"
    }
]

def main():
    # Variable for image folder path as requested
    # Default to ../../data/benchmark_images if not provided
    default_path = "data/benchmark_images"
    images_dir = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    print(f"Starting Consolidated Benchmark")
    print(f"Images Dir: {images_dir}")

    # Get list of images
    extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        # Also try uppercase
        image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    image_paths.sort()
    
    if not image_paths:
        print("No images found in directory.")
        return

    print(f"Found {len(image_paths)} images. Processing first 20...")
    
    # Process for each endpoint
    for endpoint in ENDPOINTS:
        label = endpoint["name"].upper()
        url = endpoint["url"]
        log_dir = endpoint["log_dir"]
        
        print(f"\n--- Benchmarking {label} API [{url}] ---")
        os.makedirs(log_dir, exist_ok=True)
        
        for i, img_path in enumerate(image_paths):
            if i >= 20: 
                break
                
            filename = os.path.basename(img_path)
            print(f"[{i+1}/20] Pinging {label}: {filename}")
            
            try:
                with open(img_path, 'rb') as f:
                    start_time = time.perf_counter()
                    files = {'file': f}
                    try:
                        response = requests.post(url, files=files, timeout=10) # 10s timeout
                        duration_ms = (time.perf_counter() - start_time) * 1000
    
                        log_file_path = os.path.join(log_dir, f"{filename}_log.json")
                        
                        with open(log_file_path, 'w') as log_file:
                            log_data = {
                                "filename": filename,
                                "status": response.status_code,
                                "client_latency_ms": duration_ms
                            }
                            
                            try:
                                log_data["response"] = response.json()
                            except:
                                log_data["raw_response"] = response.text
                            
                            json.dump(log_data, log_file, indent=2)
                    except requests.exceptions.RequestException as e:
                         print(f"Request failed: {e}")
                         duration_ms = (time.perf_counter() - start_time) * 1000
                         log_file_path = os.path.join(log_dir, f"{filename}_error.json")
                         with open(log_file_path, 'w') as log_file:
                             json.dump({"error": str(e), "client_latency_ms": duration_ms}, log_file)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nBenchmark complete. Logs saved to benchmark_logs/")

if __name__ == "__main__":
    main()
