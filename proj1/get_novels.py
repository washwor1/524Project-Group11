import requests
import os
from tqdm import tqdm

folder_name = 'Maurice_Leblanc_Books'
os.makedirs(folder_name, exist_ok=True)

base_url = 'https://gutendex.com/books'

params = {
    'search': 'Maurice Leblanc'
}

response = requests.get(base_url, params=params)
data = response.json()

books_to_download = []

for book in data['results']:
    if book['media_type'] != 'Text':
        continue

    exact_author_match = False
    for author in book['authors']:
        if author['name'] == 'Leblanc, Maurice':
            exact_author_match = True
            break
    if not exact_author_match:
        continue

    books_to_download.append(book)

for book in tqdm(books_to_download, desc="Downloading Books"):
    formats = book['formats']
    text_formats = [key for key in formats.keys() if key.startswith('text/plain')]
    if text_formats:
        download_url = formats[text_formats[0]]
        extension = '.txt'
    else:
        print(f"No text format available for '{book['title']}'. Skipping.")
        continue

    title = book['title']
    safe_title = ''.join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
    filename = f"{safe_title}{extension}"
    filepath = os.path.join(folder_name, filename)

    # print(f"Downloading '{title}' as {extension}...")
    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 
    with open(filepath, 'wb') as file, tqdm(
        desc=title,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

print("All books have been downloaded successfully!")
