import os
import json

def update_gifs_json():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the gifs folder
    gifs_folder = os.path.join(current_dir, 'gifs')
    
    # Get all .gif files from the folder
    gif_files = [f for f in os.listdir(gifs_folder) if f.lower().endswith('.gif')]
    
    # Create the JSON data
    data = {"gifs": gif_files}
    
    # Write the data to gifs.json in the same directory as the script
    json_file = os.path.join(current_dir, 'gifs.json')
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated gifs.json with {len(gif_files)} GIFs")

if __name__ == "__main__":
    update_gifs_json()