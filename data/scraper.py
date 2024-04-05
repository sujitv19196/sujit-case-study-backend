import csv
import re
import scrapy
from scrapy.crawler import CrawlerProcess
from w3lib.html import remove_tags, remove_tags_with_content 


class MySpider(scrapy.Spider):
    name = "my_spider"
    start_urls = ['https://www.partselect.com/']

    def __init__(self, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.visited_urls = set()
        self.isFirst = True

    async def parse(self, response):
        self.visited_urls.add(response.url)

        # Remove header, footer, script elements from the response for text parsing
        to_remove = ['noscript', 'script', 'head', 'header', 'footer', 'button', 'input', 'select', 'img']
        if self.isFirst: # keep header on home page
            to_remove = ['noscript', 'script', 'head', 'footer', 'button', 'input', 'select', 'img']
            self.isFirst = False
        response_text = remove_tags_with_content(response.text, which_ones=to_remove)
        # Parse the modified text again
        response_selector = scrapy.Selector(text=response_text)

        # Extract text from all remaining elements, including link text
        text = ' '.join(response_selector.xpath('//text() | //a//text() | //a//span//text()').extract())
        text = text.replace('\n', '').replace('\r', '').replace('\t', '')
        text = re.sub(r'\s+', ' ', text) # remove extra whitespace
        # remove coupon messages
        text = text.replace('Your coupon for will be reflected when you check out!', '')
        text = text.replace('Your coupon for has been applied and will be reflected when you check out!', '')
        text = text.replace("Hello! You're visiting the PartSelect site in U.S. Would you like to shop on the Canadian site? Stay on this site Go to Canadian site", '')
        yield {'text': text.strip(), 'url': response.url}
       
        for link in response_selector.css('a::attr(href)').extract():
            absolute_url = response.urljoin(link)
            if absolute_url.startswith('https://www.partselect.com/') and absolute_url not in self.visited_urls:
                yield scrapy.Request(absolute_url, callback=self.parse)

class CSVPipeline:
    def open_spider(self, spider):
        self.csvfile = open('data/overnight.csv', 'a', newline='')
        self.writer = csv.writer(self.csvfile)

    def close_spider(self, spider):
        self.csvfile.close()

    def process_item(self, item, spider):
        self.writer.writerow([item['text'], item['url']])
        return item

def run_spider():
    process = CrawlerProcess(settings={
        'ITEM_PIPELINES': {'__main__.CSVPipeline': 100}
    })
    process.crawl(MySpider)
    process.start()

if __name__ == "__main__":
    run_spider()
