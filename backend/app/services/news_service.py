import aiohttp
from bs4 import BeautifulSoup
from app.core.config import settings
from app.models.models import Article
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import logging
from app.services.sentiment import analyze_sentiment
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def scrape_forbes(token: str) -> list[dict]:
    """Scrape Forbes for articles about a specific token"""
    articles = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        )
        
        try:
            page = await context.new_page()
            
            # Check both news and research sections
            for section, url in {
                'news': settings.FORBES_NEWS_URL,
                'research': settings.FORBES_RESEARCH_URL
            }.items():
                
                await page.goto(url, wait_until='networkidle')
                await page.wait_for_selector('article.stream-item', timeout=10000)
                
                # Scroll to load more content
                for _ in range(3):
                    await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                    await page.wait_for_timeout(2000)
                
                # Get article links
                article_cards = await page.query_selector_all('article.stream-item')
                for card in article_cards:
                    try:
                        title_elem = await card.query_selector('a.stream-item__title')
                        if title_elem:
                            title = await title_elem.text_content()
                            url = await title_elem.get_attribute('href')
                            
                            # Check if article mentions the token
                            if token.lower() in title.lower():
                                if not url.startswith('http'):
                                    url = f"https://www.forbes.com{url}"
                                
                                # Get article content
                                await page.goto(url, wait_until='networkidle')
                                content_elem = await page.query_selector('.article-body')
                                if content_elem:
                                    content = await content_elem.text_content()
                                    articles.append({
                                        'title': title.strip(),
                                        'url': url,
                                        'content': content.strip(),
                                        'source': f'Forbes {section.capitalize()}'
                                    })
                    except Exception as e:
                        logger.error(f"Error processing article: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error scraping Forbes: {e}")
        
        finally:
            await browser.close()
    
    return articles

async def update_news_and_sentiment(db: AsyncSession):
    """Update news and sentiment data for all supported tokens"""
    try:
        for token in settings.SUPPORTED_TOKENS:
            articles = await scrape_forbes(token)
            
            for article_data in articles:
                # Check if article already exists
                result = await db.execute(
                    select(Article).where(Article.url == article_data['url'])
                )
                if result.scalar_one_or_none():
                    continue
                
                sentiment, score = await analyze_sentiment(article_data['content'], db=db)
                
                # Create and store article in database
                article = Article(
                    token=token,
                    title=article_data['title'],
                    url=article_data['url'],
                    content=article_data['content'],
                    source=article_data['source'],
                    sentiment=sentiment,
                    score=score,
                    timestamp=datetime.utcnow()
                )
                db.add(article)
            
            # Commit after processing all articles for a token
            await db.commit()
                
    except Exception as e:
        logger.error(f"Error in update_news_and_sentiment: {e}")
        await db.rollback()
        raise 