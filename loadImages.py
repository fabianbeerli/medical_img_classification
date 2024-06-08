import os
import requests

def fetch_images(num_images_to_fetch, batch_size=100):
    # Ensure num_images_to_fetch is an integer
    try:
        num_images_to_fetch = int(num_images_to_fetch)
    except ValueError:
        raise ValueError("num_images_to_fetch must be an integer.")
    
    url = 'https://api.isic-archive.com/api/v2/images'
    all_images = []

    for offset in range(0, num_images_to_fetch, batch_size):
        params = {
            'limit': min(batch_size, num_images_to_fetch - offset),
            'offset': offset
        }
        print(f"Fetching images with params: {params}")  # Debugging statement
        response = requests.get(url, params=params)
        response.raise_for_status()
        images = response.json()['results']
        all_images.extend(images)
        
        print(f"Fetched {len(images)} images, total so far: {len(all_images)}")  # Debugging statement

        if len(images) < batch_size:
            # If fewer results are returned than the batch size, it means there are no more images to fetch
            print("No more images to fetch.")
            break

    print(f"Total images fetched: {len(all_images)}")
    return all_images

def download_image(image_url, save_path, filename):
    response = requests.get(image_url)
    response.raise_for_status()
    
    # Ensure the save_path is a directory, not a file
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Check if the file already exists
    file_path = os.path.join(save_path, filename)
    index = 1
    while os.path.exists(file_path):
        # Construct the new filename with an index
        base_filename, ext = os.path.splitext(filename)
        new_filename = f"{base_filename}_{index}{ext}"
        file_path = os.path.join(save_path, new_filename)
        index += 1

    with open(file_path, 'wb') as f:
        f.write(response.content)


def main(num_images_to_fetch):
    os.makedirs('ISIC_Images', exist_ok=True)
    images = fetch_images(num_images_to_fetch)
    
    download_count = 0
    failed_count = 0
    for image in images:
        full_image_url = image['files']['full']['url']
        image_id = image['isic_id']
        
        diagnosis = image['metadata']['clinical'].get('diagnosis', 'unknown').replace(' ', '_')
        benign_malignant = image['metadata']['clinical'].get('benign_malignant', 'unknown').replace(' ', '_')
        
        # Replace '/' with '-' in benign/malignant label
        benign_malignant = benign_malignant.replace('/', '_')

        folder_path = os.path.join('ISIC_Images', f"{diagnosis}_{benign_malignant}")
        os.makedirs(folder_path, exist_ok=True)
        
        filename = image_id + '.jpg'
        
        try:
            download_image(full_image_url, folder_path, filename)
            download_count += 1
            print(f"Downloaded {image_id} to {folder_path} ({download_count}/{len(images)})")
        except requests.exceptions.HTTPError as err:
            failed_count += 1
            print(f"Failed to download {image_id}: {err}")

    print(f"Total images downloaded: {download_count}")
    print(f"Total images failed to download: {failed_count}")

if __name__ == "__main__":
    num_images_to_fetch = "1500"  # Specify the number of images you want to fetch
    main(num_images_to_fetch)
