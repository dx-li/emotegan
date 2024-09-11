import os
import json

def update_gifs_json():
    # Path to the gifs folder
    gifs_folder = 'gifs'
    
    # Get all .gif files from the folder
    gif_files = [f for f in os.listdir(gifs_folder) if f.lower().endswith('.gif')]
    
    # Create the JSON data
    data = {"gifs": gif_files}
    
    # Write the data to gifs.json
    with open('gifs.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated gifs.json with {len(gif_files)} GIFs")

if __name__ == "__main__":
    update_gifs_json()