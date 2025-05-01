import requests
from bs4 import BeautifulSoup
# from svglib.svgnode import SvgParser
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from io import BytesIO

# Define the URL of the website
url = 'https://de.m.wikipedia.org/wiki/Bildtafel_der_Verkehrszeichen_in_der_Bundesrepublik_Deutschland_seit_2017'

# Send a GET request to fetch the page content
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all <li> elements with class "gallerybox"
    gallery_items = soup.find_all('li', class_='gallerybox')
    traffic_signs = []
    
    # Iterate over the found elements and print them
    for item in gallery_items:
        img = item.find("div",class_="thumb")
        text = item.find("div",class_="gallerytext")

        img_link = img.find("a")["href"]
        textheader = text.find_all('b')#find("b")
        textdescr = text.find_all("a")
        print(textheader,textdescr)
        if(len(textheader) == 0 or len(textdescr) == 0):
            continue
        
        traffic_signs.append([img_link,textheader[0].text,textdescr[0].text])
        
        print(img_link,textheader,textdescr)
    print("d")

svg_url = traffic_signs[0][0]

# Send a GET request to download the SVG file
response = requests.get("https://de.m.wikipedia.org/" + svg_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the SVG content
    svg_data = BytesIO(response.content)
    # drawing = SvgParser.parse(svg_data)
    drawing = svg2rlg(svg_data)
    
    # Convert to PNG using reportlab's renderPM
    # output_path = 'output_image.png'
    # renderPM.drawToFile(drawing, output_path, fmt='PNG')
    renderPM.drawToFile(drawing, "file.png", fmt="PNG")

    # print(f"Conversion complete. PNG saved as '{output_path}'.")
else:
    print(f"Failed to download the SVG. Status code: {response.status_code}")