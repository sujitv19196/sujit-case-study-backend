import csv
import re
import scrapy
from scrapy.crawler import CrawlerProcess
from w3lib.html import remove_tags_with_content 
import sys
class MySpider(scrapy.Spider):
    name = "my_spider"
    start_urls = ['https://www.partselect.com/Dishwasher-Parts.htm', 'https://www.partselect.com/Refrigerator-Parts.htm']

    def __init__(self, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.visited_urls = set()
        self.isFirst = True

    async def parse(self, response):
        self.visited_urls.add(response.url)

        # Remove header, footer, script elements from the response for text parsing
        to_remove = ['noscript', 'script', 'head', 'header', 'footer', 'button', 'input' 'select', 'img']
        response_text = remove_tags_with_content(response.text, which_ones=to_remove)
        
        # Remove input elements and their content from the response for text parsing
        response_text = re.sub(r'<input[^>]*>', '', response_text)

        # Parse the modified text again
        response_selector = scrapy.Selector(text=response_text)

        # Extract text from all remaining elements, including link text
        text = ' '.join(response_selector.xpath('//text() | //a//text() | //a//span//text()').extract())
        # text = text.replace('\n', '').replace('\r', '').replace('\t', '')
        text = re.sub(r'\s+', ' ', text) # remove extra whitespace
        # remove coupon messages
        text = text.replace('Your coupon for will be reflected when you check out!', '')
        text = text.replace('Your coupon for has been applied and will be reflected when you check out!', '')
        text = text.replace("Hello! You're visiting the PartSelect site in U.S. Would you like to shop on the Canadian site? Stay on this site Go to Canadian site", '')

        title = response.css('title::text').get()
        model_num = await self.find_model_num(title)
        ps_num = await self.find_part_select_num(response.url)
        yield {'text': text.strip(), 'url': response.url, 'depth': response.meta["depth"], 'title': title, 'model_num': model_num, 'ps_num': ps_num}
        for link in response_selector.css('a::attr(href)').extract():
            if (link != "/ShoppingCart.aspx"):
                absolute_url = response.urljoin(link)
                if absolute_url != 'https://www.partselect.com' and absolute_url.startswith('https://www.partselect.com/') and absolute_url not in self.visited_urls:
                    yield scrapy.Request(absolute_url, callback=self.parse)
    
    async def find_model_num(self, title):
        max_number_count = 0
        part_num = "N/A"
        
        # Split the title into words and iterate over them
        for word in title.split():
            # Count the number of digits in the word
            number_count = sum(1 for char in word if char.isdigit())
            if number_count > max_number_count:
                max_number_count = number_count
                part_num = word.strip()
        
        return part_num
    
    async def find_part_select_num(self, url):
        # Regular expression to match "PS" followed by digits
        pattern = r'PS\d+'
        match = re.search(pattern, url)
        if match:
            return match.group(0)  # Return the matched part select number
        else:
            return "N/A"

class CSVPipeline:
    def open_spider(self, spider):
        self.csvfile = open(sys.argv[1], 'a', newline='')
        self.writer = csv.writer(self.csvfile)

    def close_spider(self, spider):
        self.csvfile.close()

    def process_item(self, item, spider):
        self.writer.writerow([item['text'], item['url'], item['depth'], item['title'], item['model_num'], item['ps_num']])
        return item

def run_spider():
    process = CrawlerProcess(settings={
        'DEPTH_LIMIT': 7,
        'DEPTH_PRIORITY': 2,
        'LOG_LEVEL': 'INFO',
        'ITEM_PIPELINES': {'__main__.CSVPipeline': 100}
    })
    process.crawl(MySpider)
    process.start()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the name of the CSV file as a command-line argument.")
    else:
        run_spider()
