import os
from PIL import Image
import numpy as np

def process_gif(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            # Check if the GIF is 56x56
            if img.size != (56, 56):
                return False
            
            # Get all frames
            frames = []
            durations = []
            for frame in range(0, img.n_frames):
                img.seek(frame)
                # Convert to RGBA to ensure consistency
                frames.append(img.convert("RGBA"))
                durations.append(img.info.get('duration', 100))  # Default to 100ms if duration not specified
            
            # Process frames
            if len(frames) < 16:
                # Repeat frames if less than 16
                repeat_count = 16 // len(frames)
                remainder = 16 % len(frames)
                frames = frames * repeat_count + frames[:remainder]
                durations = durations * repeat_count + durations[:remainder]
            elif len(frames) > 16:
                # Downsample to 16 frames
                step = len(frames) / 16
                frames = [frames[int(i * step)] for i in range(16)]
                durations = [durations[int(i * step)] for i in range(16)]
            
            # Calculate average duration
            avg_duration = sum(durations) // len(durations)
            
            # Save the processed GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=avg_duration,
                loop=0,
                disposal=2  # Clear the frame before rendering the next
            )
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    input_folder = 'twitch_emotes'
    output_folder = 'processed_gifs'
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.gif'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            if process_gif(input_path, output_path):
                print(f"Processed: {filename}")
            else:
                print(f"Skipped: {filename}")

if __name__ == "__main__":
    main()