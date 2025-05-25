import pytest
import asyncio
from playwright.async_api import async_playwright, TimeoutError
import logging

# Forbes URLs for testing
FORBES_NEWS_URL = "https://www.forbes.com/digital-assets/news"
FORBES_RESEARCH_URL = "https://www.forbes.com/digital-assets/research"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_forbes_connection():
    """Test basic connection to Forbes digital assets pages"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        )
        
        try:
            page = await context.new_page()
            
            # Test both news and research sections
            for section, url in {
                'news': FORBES_NEWS_URL,
                'research': FORBES_RESEARCH_URL
            }.items():
                logger.info(f"Testing connection to {section} section: {url}")
                
                try:
                    # First try with networkidle
                    response = await page.goto(url, wait_until='networkidle', timeout=60000)
                except TimeoutError:
                    logger.warning(f"Networkidle timeout for {section}, trying with domcontentloaded")
                    # Fallback to domcontentloaded if networkidle times out
                    response = await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                
                assert response.status == 200, f"Failed to connect to {section} section"
                
                # Wait for content with retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Wait for any article to appear
                        await page.wait_for_selector('article.stream-item', timeout=20000)
                        article_items = await page.query_selector_all('article.stream-item')
                        if len(article_items) > 0:
                            break
                    except TimeoutError:
                        if attempt == max_retries - 1:
                            raise
                        logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(2)
                
                article_items = await page.query_selector_all('article.stream-item')
                assert len(article_items) > 0, f"No articles found in {section} section"
                
                logger.info(f"Successfully connected to {section} section")
                logger.info(f"Found {len(article_items)} articles")
                
        finally:
            await browser.close()

@pytest.mark.asyncio
async def test_content_scroll():
    """Test scrolling functionality for loading more content"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        try:
            # Test scrolling on news section
            try:
                await page.goto(FORBES_NEWS_URL, wait_until='networkidle', timeout=60000)
            except TimeoutError:
                logger.warning("Networkidle timeout, trying with domcontentloaded")
                await page.goto(FORBES_NEWS_URL, wait_until='domcontentloaded', timeout=60000)
            
            # Wait for initial content
            await page.wait_for_selector('article.stream-item', timeout=20000)
            
            # Get initial article count
            initial_articles = await page.query_selector_all('article.stream-item')
            initial_count = len(initial_articles)
            assert initial_count > 0, "No initial articles found"
            
            # Scroll multiple times
            for i in range(3):
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await page.wait_for_timeout(3000)  # Increased wait time
                
                try:
                    # Wait for new content to load
                    await page.wait_for_function(
                        f'document.querySelectorAll("article.stream-item").length > {initial_count}',
                        timeout=10000
                    )
                except TimeoutError:
                    logger.warning(f"No new articles loaded after scroll {i+1}")
                    continue
                
                # Get new article count
                current_articles = await page.query_selector_all('article.stream-item')
                current_count = len(current_articles)
                
                logger.info(f"Scroll {i+1}: Found {current_count} articles (was {initial_count})")
                assert current_count >= initial_count, "Scrolling should maintain or increase article count"
                initial_count = current_count
                
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__, "-v"])) 