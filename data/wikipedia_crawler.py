import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json

URL = "https://fr.wikipedia.org/w/api.php"
RESULT_FILE = "data/data.txt"
METADATA_FILE = "data/metadata.json"
ENCODING = "utf-8"
START_PAGE = "Intelligence_artificielle"
MAX_WORKERS = os.cpu_count()
MAX_QUEUE = 10_000
UPDATE_METADATA_FREQUENCIE = 10
# OBJECTIVE = float("inf")
OBJECTIVE = 1000


class WikipediaCrawler:
    def __init__(self):
        self.nb_visited_articles = 0
        self.visited_articles = set()
        self.queue = []
        self.file_result_con = open(RESULT_FILE, "a", encoding=ENCODING)

        self.S = requests.Session()

    def launch(self, objective, load_meta=False):
        if load_meta:
            self.load_metadata()
            print(f"visited:{self.nb_visited_articles}")
            print(f"queue:{len(self.queue)}")
        else:
            self.queue.append(START_PAGE)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            while self.nb_visited_articles < objective:
                while self.queue and len(futures) < MAX_WORKERS:
                    next_subject = self.queue.pop(0)
                    if next_subject not in self.visited_articles:
                        self.visited_articles.add(next_subject)
                        self.nb_visited_articles += 1
                        futures.append(executor.submit(self.process_page, next_subject))

                        if self.nb_visited_articles % UPDATE_METADATA_FREQUENCIE == 0:
                            self.save_metadata()
                for future in as_completed(futures):
                    links = future.result()
                    self.queue.extend([l for l in links if l not in self.visited_articles])
                    self.queue = self.queue[:MAX_QUEUE]
                    futures.remove(future)
            self.save_metadata()
            self.file_result_con.close()

    def process_page(self, page):
        html_content, links = self.get_wikipedia_content(page)
        if html_content:
            text_content = self.html_to_text(html_content)
            clean_text_content = self.clean_text(text_content)
            self.save_content(clean_text_content, page)
        print(f"subject:{page}   {self.nb_visited_articles}  added_links:{len(links)}")
        return links

    def get_wikipedia_content(self, page):
        PARAMS = {
            "action": "parse",
            "page": page,
            "format": "json",
            "prop": "text|links"
        }
        try:
            response = self.S.get(url=URL, params=PARAMS, timeout=10)
            response.raise_for_status()
            data = response.json()
            html_content = data['parse']['text']['*']
            links = [link['*'] for link in data['parse']['links'] if link['ns'] == 0]
            return html_content, links
        except requests.RequestException as e:
            print(f"Erreur lors de la requête HTTP : {e}")
            return "", []
        except KeyError:
            print("Réponse inattendue de l'API.")
            return "", []

    def save_content(self, content, subject):
        self.file_result_con.write(f"***{subject}***\n{content}\n")

    def save_metadata(self):
        data = {
            "nb_articles": len(self.visited_articles),
            "artciles": list(self.visited_articles),
            "queue":self.queue
        }
        with open(METADATA_FILE, "w+") as f:
            json.dump(data, f, indent=2)
        f.close()

    def load_metadata(self):
        with open(METADATA_FILE, "r", encoding=ENCODING) as f:
            data = json.load(f)
        if data:
            self.visited_articles = set(data.get("artciles"))
            self.nb_visited_articles = len(self.visited_articles)
            self.queue = data.get("queue")

    @staticmethod
    def html_to_text(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for math in soup.find_all('math'):
            math.decompose()
        text_parts = soup.find_all('p')
        content_text = ' '.join(part.get_text() for part in text_parts)
        return content_text

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\xa0', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = text.replace("-", "")
        return text


if __name__ == "__main__":
    print(f"threads:{MAX_WORKERS}")
    wc = WikipediaCrawler()
    wc.launch(OBJECTIVE, load_meta=False)
