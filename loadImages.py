import os
import re
import requests

def fetch_images(num_images_to_fetch=10):
    url = 'https://api.isic-archive.com/api/v2/images'
    params = {
        'limit': num_images_to_fetch,
        'offset': 0
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def download_image(image_url, save_path, filename):
    response = requests.get(image_url)
    response.raise_for_status()
    
    save_path = os.path.join(save_path, filename)
    with open(save_path, 'wb') as f:
        f.write(response.content)

def main(num_images_to_fetch):
    os.makedirs('ISIC_Images', exist_ok=True)
    images = fetch_images(num_images_to_fetch)
    for image in images['results']:
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
            print(f"Downloaded {image_id} to {folder_path}")
        except requests.exceptions.HTTPError as err:
            print(f"Failed to download {image_id}: {err}")

if __name__ == "__main__":
    main()
