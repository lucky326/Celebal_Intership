import requests
from collections import Counter
import re
from bs4 import BeautifulSoup

url = "http://www.poornima.org/"

# Fetch the webpage content
response = requests.get(url)
html_content = response.text

# Fetching hyperlinks 
soup = BeautifulSoup(html_content, 'html.parser')
links = [a['href'] for a in soup.find_all('a', href=True)]

#frequency
link_counter = Counter(links)

# Rank the hyperlinks based on their frequency
ranked_links = link_counter.most_common()

for link, frequency in ranked_links:
    print(f"{link}: {frequency}\n")