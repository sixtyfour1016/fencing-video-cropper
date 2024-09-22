import requests
from bs4 import BeautifulSoup
import os
import re

# Base URL for searching clips
base_url = "https://fencingdatabase.com/"
search_url = ("https://fencingdatabase.com/?firstname=Junho&lastname=&weapon=all&gender=all"
              "&tournament=all&year=all&opposing-score=0&opposing-lastname=&submit-search=Search%20Clips&page=")

# Folder to save the downloaded videos
output_folder = "C:/Users/STT015/PycharmProjects/fencing-video-cropper/input_videos"
os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists


def get_smallest_available_index(folder):
    """
    Find the smallest available index by scanning existing filenames and choosing the first gap.
    """
    existing_indices = []
    file_pattern = re.compile(r"fencing_clip_(\d+)\.mp4")

    # Get the list of all files in the folder
    for file_name in os.listdir(folder):
        match = file_pattern.match(file_name)
        if match:
            existing_indices.append(int(match.group(1)))

    # Find the smallest available index
    index = 1
    while index in existing_indices:
        index += 1

    return index


def video_already_downloaded(url):
    """
    Check if the video has already been downloaded by comparing video URLs.
    """
    downloaded_videos_file = os.path.join(output_folder, "downloaded_videos.txt")

    # If the file doesn't exist, return False (video hasn't been downloaded)
    if not os.path.exists(downloaded_videos_file):
        return False

    # Read the file and check if the URL is present
    with open(downloaded_videos_file, "r") as file:
        downloaded_urls = file.read().splitlines()

    return url in downloaded_urls


def save_video_url(url):
    """
    Save the video URL to a file so it isn't downloaded again.
    """
    downloaded_videos_file = os.path.join(output_folder, "downloaded_videos.txt")
    with open(downloaded_videos_file, "a") as file:
        file.write(url + "\n")


def download_videos_from_page(page_url):
    """
    Download all videos from a given page URL.
    """
    response = requests.get(page_url)
    soup = BeautifulSoup(response.content, "html.parser")
    videos = soup.find_all("div", class_="video")

    for video in videos:
        source = video.find("source")
        if source and "src" in source.attrs:
            video_url = source["src"]

            # Check if the video has already been downloaded
            if video_already_downloaded(video_url):
                print(f"Skipping already downloaded video: {video_url}")
                continue

            # Get the smallest available index
            smallest_index = get_smallest_available_index(output_folder)

            # Create the filename with the smallest available index
            video_name = f"fencing_clip_{smallest_index}.mp4"

            # Download the video
            print(f"Downloading {video_name} from {video_url}...")
            video_response = requests.get(video_url)

            # Save the video to the output folder
            video_path = os.path.join(output_folder, video_name)
            with open(video_path, "wb") as f:
                f.write(video_response.content)

            print(f"Downloaded and saved as: {video_name}")

            # Save the video URL so it won't be downloaded again
            save_video_url(video_url)


def get_next_page_url(soup):
    """
    Check if there's a 'Next' page button in the 'touches-pagination' div and return its URL.
    """
    pagination = soup.find("div", class_="touches-pagination")
    if pagination:
        next_page_button = pagination.find("a", text="Next")
        if next_page_button and 'href' in next_page_button.attrs:
            return base_url + next_page_button['href']
    return None


# Start downloading videos from the first page
current_page = 1
current_page_url = search_url + str(current_page)

while current_page_url:
    print(f"Processing page {current_page}...")

    # Request the page and parse it
    response = requests.get(current_page_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Download all videos from the current page
    download_videos_from_page(current_page_url)

    # Try to find the URL for the next page
    next_page_url = get_next_page_url(soup)

    # Update the page URL for the next iteration (if there's a next page)
    if next_page_url:
        current_page_url = next_page_url
        current_page += 1
    else:
        current_page_url = None

print("All videos have been downloaded.")
