import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
import re
from playwright.async_api import async_playwright, Page, Browser, TimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForbesScraper:
    def __init__(self):
        self.sections = {
            'news': "https://www.forbes.com/digital-assets/news",
            'research': "https://www.forbes.com/digital-assets/research"
        }

    async def clean_text(self, text: str) -> str:
        """Clean up text by removing extra whitespace and unwanted characters"""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'[\n\r\t]', '', text)  # Remove newlines, carriage returns, tabs
        return text.strip()

    async def extract_article_content(self, page: Page, url: str) -> Optional[Dict]:
        """Extract content from a Forbes article page"""
        try:
            # Navigate to article and wait for content
            await page.goto(url, wait_until='networkidle')
            await page.wait_for_load_state('domcontentloaded')
            
            # Wait for key elements with appropriate timeout
            try:
                await page.wait_for_selector('article', timeout=10000)
            except TimeoutError:
                logger.warning(f"Timeout waiting for article content at {url}")
                return None

            # Extract article components using Forbes' selectors
            title_elem = await page.query_selector('h1[data-ga-track="headline"]')
            if not title_elem:
                title_elem = await page.query_selector('.article-headline')  # Fallback selector
            
            author_elem = await page.query_selector('.fs-author-name')
            date_elem = await page.query_selector('.fs-article-date')
            content_elem = await page.query_selector('.article-body')

            if title_elem and content_elem:
                # Extract text content
                title_text = await self.clean_text(await title_elem.text_content())
                author_text = "Unknown"
                if author_elem:
                    author_text = await self.clean_text(await author_elem.text_content())
                date_text = ""
                if date_elem:
                    date_text = await self.clean_text(await date_elem.text_content())

                # Get article paragraphs
                paragraphs = await content_elem.query_selector_all('p, h2, h3')
                content_texts = []
                for p in paragraphs:
                    text = await p.text_content()
                    if text:
                        content_texts.append(await self.clean_text(text))
                content_text = " ".join(content_texts)

                return {
                    'title': title_text,
                    'author': author_text,
                    'published': date_text,
                    'link': url,
                    'summary': content_text[:500] + '...' if len(content_text) > 500 else content_text,
                    'full_content': content_text
                }

            return None
        except Exception as e:
            logger.error(f"Error extracting article content from {url}: {e}")
            return None

    async def scroll_and_collect_articles(self, page: Page, section: str) -> List[Dict]:
        """Scroll through the page and collect all visible article links"""
        article_links = set()  # Use set to avoid duplicates
        
        try:
            # Initial wait for content
            await page.wait_for_selector('article.stream-item', timeout=10000)
            
            # Scroll multiple times to load more content
            for _ in range(5):  # Adjust number of scrolls as needed
                # Get current articles
                cards = await page.query_selector_all('article.stream-item')
                for card in cards:
                    try:
                        # Get headline link
                        headline = await card.query_selector('a.stream-item__title')
                        if headline:
                            url = await headline.get_attribute('href')
                            if url:
                                if not url.startswith('http'):
                                    url = f"https://www.forbes.com{url}"
                                article_links.add(url)
                                
                                # Log found article
                                title = await headline.text_content()
                                logger.info(f"\nFound {section} article:")
                                logger.info(f"Title: {await self.clean_text(title)}")
                                logger.info(f"Link: {url}")
                    except Exception as e:
                        logger.error(f"Error processing article card: {e}")
                        continue
                
                # Scroll down
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await asyncio.sleep(2)  # Wait for new content to load
                
                # Check if we've reached the bottom
                old_height = await page.evaluate('document.body.scrollHeight')
                await asyncio.sleep(2)
                new_height = await page.evaluate('document.body.scrollHeight')
                if old_height == new_height:
                    break
        
        except Exception as e:
            logger.error(f"Error during scrolling and collection: {e}")
        
        return list(article_links)

    async def fetch_articles(self, limit: int = 5) -> Dict[str, List[Dict]]:
        """Fetch latest articles from both news and research sections"""
        results = {'news': [], 'research': []}
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
            )
            
            try:
                page = await context.new_page()
                
                # Process each section
                for section, url in self.sections.items():
                    logger.info(f"\nProcessing {section} section...")
                    
                    # Navigate to section page
                    await page.goto(url, wait_until='networkidle')
                    
                    # Collect article links
                    article_links = await self.scroll_and_collect_articles(page, section)
                    logger.info(f"Found {len(article_links)} {section} articles")
                    
                    # Process articles up to the limit
                    for url in article_links[:limit]:
                        article = await self.extract_article_content(page, url)
                        if article:
                            results[section].append(article)
                            
                            # Print article details
                            logger.info(f"\nExtracted {section} article:")
                            logger.info(f"Title: {article['title']}")
                            logger.info(f"Author: {article['author']}")
                            logger.info(f"Published: {article['published']}")
                            logger.info(f"Link: {article['link']}")
                            logger.info(f"Summary: {article['summary'][:200]}...")
                
                # Save results to separate files
                for section, articles in results.items():
                    filename = f'forbes_{section}_articles.json'
                    with open(filename, 'w') as f:
                        json.dump(articles, f, indent=2)
                    logger.info(f"\nSaved {len(articles)} {section} articles to {filename}")
                
            except Exception as e:
                logger.error(f"Error during Forbes scraping: {e}")
                logger.error(f"Error details: {str(e)}")
            
            finally:
                await browser.close()
        
        return results

async def main():
    scraper = ForbesScraper()
    await scraper.fetch_articles()

if __name__ == "__main__":
    asyncio.run(main()) 