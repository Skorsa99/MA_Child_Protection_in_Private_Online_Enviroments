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

def log_duplicate_remove(message):
    # Ensure the directory exists
    file_path = 'logs/duplicate_remove.txt'
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

def log_data_filtering(message):
    # Ensure the directory exists
    file_path = 'logs/data_filtering.txt'
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

def log_data_manifest(message):
    # Ensure the directory exists
    file_path = 'logs/manifest_creation.txt'
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

def log_data_training(message):
    # Ensure the directory exists
    file_path = 'logs/manifest_creation.txt'
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
    
def image_tally(number_images, category):
    # Ensure the directory exists
    file_path = 'logs/image_tally.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Read existing data
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {f"image_count_{category}": 0}

    # Update the image count
    key = f"image_count_{category}"
    data[key] = data.get(key, 0) + number_images

    # Write the updated data back to the file
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
    except Exception as e:
        print(f"Error writing to image tally file: {e}")

    updated_tally = data[f"image_count_{category}"]
    return updated_tally