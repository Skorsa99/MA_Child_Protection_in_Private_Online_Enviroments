import os
import json
import datetime

def log_data_collection(message):
    # Ensure the directory exists
    file_path = 'logs/data_collection.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Get a date-time string and format the log message
    date_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"{date_time_str} | {message}\n"

    # Open the log file in append mode and write the message
    try:
        with open(file_path, 'a') as log_file:
            log_file.write(full_message)
    except Exception as e:
        return
    
def image_tally(number_images):
    # Ensure the directory exists
    file_path = 'logs/image_tally.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Read existing data
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"image_count": 0}

    # Update the image count
    data["image_count"] += number_images

    # Write the updated data back to the file
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
    except Exception as e:
        print(f"Error writing to image tally file: {e}")
    
    updated_tally = data["image_count"]
    return updated_tally