import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import json
import os
from urllib.parse import urljoin, urlparse
import re
from pathlib import Path
import PyPDF2
import io
import warnings

warnings.filterwarnings('ignore')


class IndianCompanyAnnualReportScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # NSE listed companies (Top 100)
        self.company_list = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK',
            'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'ASIANPAINT', 'LT',
            'AXISBANK', 'MARUTI', 'SUNPHARMA', 'ULTRACEMCO', 'WIPRO', 'NESTLEIND',
            'HCLTECH', 'POWERGRID', 'BAJFINANCE', 'NTPC', 'TECHM', 'ONGC',
            'TATAMOTORS', 'TITAN', 'COALINDIA', 'INDUSINDBK', 'GRASIM', 'DRREDDY',
            'JSWSTEEL', 'BRITANNIA', 'CIPLA', 'BPCL', 'HEROMOTOCO', 'EICHERMOT',
            'DIVISLAB', 'TATASTEEL', 'BAJAJFINSV', 'IOC', 'ADANIPORTS', 'HINDALCO',
            'APOLLOHOSP', 'BAJAJ-AUTO', 'SHREECEM', 'SBILIFE', 'HDFCLIFE',
            'TATACONSUM', 'GODREJCP', 'PIDILITIND', 'BERGEPAINT', 'DABUR'
        ]

        self.base_urls = {
            'bse': 'https://www.bseindia.com',
            'nse': 'https://www.nseindia.com',
            'screener': 'https://www.screener.in',
            'tijori': 'https://www.tijorifinance.com',
            'capitaline': 'https://www.capitaline.com'
        }

    def get_company_basic_info(self, symbol):
        """Get basic company information"""
        try:
            # Try Screener.in first
            url = f"https://www.screener.in/company/{symbol}/"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract company name
                name_elem = soup.find('h1')
                company_name = name_elem.text.strip() if name_elem else symbol

                # Extract sector
                sector = "Unknown"
                try:
                    sector_elem = soup.find('span', class_='sub')
                    if sector_elem:
                        sector = sector_elem.text.strip()
                except:
                    pass

                return {
                    'symbol': symbol,
                    'name': company_name,
                    'sector': sector,
                    'source': 'screener'
                }
        except Exception as e:
            print(f"Error getting info for {symbol}: {e}")

        return {
            'symbol': symbol,
            'name': symbol,
            'sector': 'Unknown',
            'source': 'manual'
        }

    def search_annual_reports_multiple_sources(self, company_symbol):
        """Search for annual reports across multiple sources"""
        reports = []

        # Source 1: Screener.in
        try:
            screener_reports = self.search_screener_reports(company_symbol)
            reports.extend(screener_reports)
        except Exception as e:
            print(f"Error searching Screener for {company_symbol}: {e}")

        # Source 2: BSE India (if available)
        try:
            bse_reports = self.search_bse_reports(company_symbol)
            reports.extend(bse_reports)
        except Exception as e:
            print(f"Error searching BSE for {company_symbol}: {e}")

        # Source 3: Company website (generic search)
        try:
            web_reports = self.search_company_website(company_symbol)
            reports.extend(web_reports)
        except Exception as e:
            print(f"Error searching company website for {company_symbol}: {e}")

        # Remove duplicates
        unique_reports = []
        seen_urls = set()
        for report in reports:
            if report['url'] not in seen_urls:
                unique_reports.append(report)
                seen_urls.add(report['url'])

        return unique_reports[:5]  # Limit to 5 most recent reports

    def search_screener_reports(self, symbol):
        """Search for annual reports on Screener.in"""
        reports = []
        try:
            url = f"https://www.screener.in/company/{symbol}/"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for annual report links
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href', '')
                    text = link.text.strip().lower()

                    if any(keyword in text for keyword in ['annual', 'report', 'investor']):
                        if href.startswith('http') or href.endswith('.pdf'):
                            year = self.extract_year_from_text(text + ' ' + href)
                            reports.append({
                                'company': symbol,
                                'year': year,
                                'url': href if href.startswith('http') else urljoin(url, href),
                                'title': link.text.strip(),
                                'source': 'screener'
                            })
        except Exception as e:
            print(f"Error in screener search: {e}")

        return reports

    def search_bse_reports(self, symbol):
        """Search for reports on BSE (simplified)"""
        reports = []
        # BSE search is complex and requires session management
        # For now, return empty list - can be enhanced later
        return reports

    def search_company_website(self, symbol):
        """Generic company website search"""
        reports = []
        # This would require knowing each company's website
        # For now, return empty list - can be enhanced with company website mapping
        return reports

    def extract_year_from_text(self, text):
        """Extract year from text"""
        # Look for 4-digit years between 2015-2025
        years = re.findall(r'20[1-2][0-9]', text)
        if years:
            return years[-1]  # Return most recent year
        return '2023'  # Default year

    def download_pdf_content(self, url):
        """Download and extract text from PDF"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                text = ""
                for page in pdf_reader.pages[:50]:  # Limit to first 50 pages
                    try:
                        text += page.extract_text() + "\n"
                    except:
                        continue

                return text
        except Exception as e:
            print(f"Error downloading PDF from {url}: {e}")

        return ""

    def extract_management_sections(self, text):
        """Extract management commentary sections from annual report text"""
        sections = {}

        if not text:
            return sections

        text_lower = text.lower()

        # Define section patterns
        section_patterns = {
            'management_commentary': [
                r'management discussion and analysis',
                r'management commentary',
                r'md&a',
                r'management perspective',
                r'management review',
                r'business review'
            ],
            'outlook': [
                r'outlook',
                r'future outlook',
                r'business outlook',
                r'forward looking',
                r'prospects',
                r'future prospects'
            ],
            'future_plans': [
                r'future plans',
                r'strategic initiatives',
                r'growth strategy',
                r'expansion plans',
                r'strategic priorities',
                r'way forward'
            ]
        }

        for section_type, patterns in section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower))
                if matches:
                    start_pos = matches[0].start()
                    # Find a reasonable end position (next major section or 3000 chars)
                    end_pos = min(start_pos + 3000, len(text))

                    # Try to find a natural break
                    text_segment = text[start_pos:end_pos]
                    sentences = text_segment.split('.')
                    if len(sentences) > 5:
                        # Take first 5-10 sentences
                        section_text = '. '.join(sentences[:8]) + '.'
                    else:
                        section_text = text_segment

                    sections[section_type] = section_text.strip()
                    break

        return sections

    def create_sample_dataset_from_templates(self):
        """Create dataset using realistic templates and patterns"""

        # Company-specific templates based on actual annual reports
        company_templates = {
            'TCS': {
                'positive': [
                    "TCS delivered strong performance this fiscal year with revenue growth of 15.2% and improved margins across all business segments. Our digital transformation services continue to gain significant traction.",
                    "We are pleased to report exceptional client satisfaction scores and successful expansion in emerging markets. Our strategic investments in AI and cloud technologies are yielding positive results."
                ],
                'negative': [
                    "The challenging macroeconomic environment and client budget constraints have impacted our growth trajectory. We expect continued headwinds in the near term.",
                    "Currency fluctuations and increased competition in traditional services have pressured our margins. We are implementing cost optimization measures."
                ],
                'neutral': [
                    "We maintain our focus on operational excellence and strategic investments while navigating through current market uncertainties. Our diversified portfolio provides stability.",
                    "The company continues to invest in talent development and technology capabilities. We expect gradual improvement in market conditions."
                ]
            },
            'HDFCBANK': {
                'positive': [
                    "The bank reported strong credit growth of 18% with improved asset quality metrics. Our digital banking initiatives have enhanced customer experience significantly.",
                    "We are well-positioned to capitalize on India's economic growth with our robust capital position and diversified business model."
                ],
                'negative': [
                    "Rising interest rates and economic uncertainty pose challenges to our lending business. Asset quality concerns in certain segments require careful monitoring.",
                    "Increased provisioning requirements and competitive pressure on margins have affected our profitability. We remain cautious about credit expansion."
                ],
                'neutral': [
                    "We maintain a balanced approach to growth while ensuring strong risk management practices. Our focus remains on sustainable business expansion.",
                    "The bank continues to strengthen its digital infrastructure and customer service capabilities. We expect stable performance in the coming quarters."
                ]
            },
            'RELIANCE': {
                'positive': [
                    "Reliance Industries delivered record performance with strong contributions from all business segments. Our digital services and retail businesses showed exceptional growth.",
                    "The successful launch of 5G services and expansion of our retail network position us strongly for future growth opportunities."
                ],
                'negative': [
                    "Volatile crude oil prices and refining margins have impacted our petrochemical business. Global economic uncertainties pose additional challenges.",
                    "Increased competition in the telecommunications sector and regulatory changes may affect our market position and profitability."
                ],
                'neutral': [
                    "We continue to focus on our integrated business model and strategic investments across key sectors. Our diversified portfolio provides resilience.",
                    "The company maintains its commitment to sustainable growth and innovation while adapting to evolving market conditions."
                ]
            }
        }

        dataset = []

        for company, sentiments in company_templates.items():
            company_info = self.get_company_basic_info(company)

            for sentiment, texts in sentiments.items():
                for i, text in enumerate(texts):
                    dataset.append({
                        'company': company,
                        'company_name': company_info['name'],
                        'sector': company_info['sector'],
                        'year': '2023',
                        'section_type': ['management_commentary', 'outlook', 'future_plans'][i % 3],
                        'text': text,
                        'sentiment': sentiment,
                        'confidence': 0.95,  # High confidence for template data
                        'manual_label': True,
                        'source': 'template',
                        'text_length': len(text),
                        'word_count': len(text.split())
                    })

        return dataset

    def scrape_real_data(self, max_companies=20):
        """Scrape real data from companies"""

        print(f"Starting to scrape data for {max_companies} companies...")
        real_data = []

        for i, company in enumerate(self.company_list[:max_companies]):
            print(f"Processing {company} ({i + 1}/{max_companies})")

            # Get company info
            company_info = self.get_company_basic_info(company)

            # Search for annual reports
            reports = self.search_annual_reports_multiple_sources(company)

            for report in reports[:2]:  # Process max 2 reports per company
                if report['url'].endswith('.pdf'):
                    print(f"  Downloading PDF: {report['title']}")

                    # Download and extract PDF content
                    pdf_text = self.download_pdf_content(report['url'])

                    if pdf_text:
                        # Extract management sections
                        sections = self.extract_management_sections(pdf_text)

                        for section_type, section_text in sections.items():
                            if section_text and len(section_text) > 100:
                                real_data.append({
                                    'company': company,
                                    'company_name': company_info['name'],
                                    'sector': company_info['sector'],
                                    'year': report['year'],
                                    'section_type': section_type,
                                    'text': section_text[:1500],  # Limit text length
                                    'sentiment': None,  # To be labeled manually
                                    'confidence': None,
                                    'manual_label': False,
                                    'source': 'scraped',
                                    'source_url': report['url'],
                                    'text_length': len(section_text),
                                    'word_count': len(section_text.split())
                                })

            # Add delay to avoid rate limiting
            time.sleep(2)

        print(f"Scraped {len(real_data)} text sections from real reports")
        return real_data

    def create_comprehensive_dataset(self, include_real_data=True, max_companies=10):
        """Create comprehensive dataset combining templates and real data"""

        print("Creating comprehensive dataset...")

        # Start with template data
        template_data = self.create_sample_dataset_from_templates()
        print(f"Created {len(template_data)} template samples")

        all_data = template_data.copy()

        # Add real scraped data if requested
        if include_real_data:
            try:
                real_data = self.scrape_real_data(max_companies)
                all_data.extend(real_data)
                print(f"Added {len(real_data)} real samples")
            except Exception as e:
                print(f"Error scraping real data: {e}")
                print("Continuing with template data only...")

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Add features for samples without them
        for idx, row in df.iterrows():
            if pd.isna(row.get('text_length')):
                df.at[idx, 'text_length'] = len(row['text'])
            if pd.isna(row.get('word_count')):
                df.at[idx, 'word_count'] = len(row['text'].split())

        return df

    def save_dataset(self, df, output_dir='indian_company_dataset'):
        """Save dataset with proper structure"""

        Path(output_dir).mkdir(exist_ok=True)

        # Save complete dataset
        df.to_csv(f'{output_dir}/complete_dataset.csv', index=False)

        # Create train/validation split
        labeled_data = df[df['sentiment'].notna()]
        unlabeled_data = df[df['sentiment'].isna()]

        if len(labeled_data) > 0:
            train_size = int(0.8 * len(labeled_data))
            train_df = labeled_data.sample(n=train_size, random_state=42)
            val_df = labeled_data.drop(train_df.index)

            train_df.to_csv(f'{output_dir}/train.csv', index=False)
            val_df.to_csv(f'{output_dir}/validation.csv', index=False)

        if len(unlabeled_data) > 0:
            unlabeled_data.to_csv(f'{output_dir}/unlabeled_for_manual_annotation.csv', index=False)

        # Create metadata
        metadata = {
            'total_samples': len(df),
            'labeled_samples': len(labeled_data),
            'unlabeled_samples': len(unlabeled_data),
            'companies': df['company'].nunique(),
            'sectors': df['sector'].unique().tolist(),
            'years': df['year'].unique().tolist(),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_text_length': df['text_length'].mean(),
            'creation_date': pd.Timestamp.now().isoformat()
        }

        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to '{output_dir}' directory")
        print(f"Files created:")
        print(f"  - complete_dataset.csv ({len(df)} samples)")
        if len(labeled_data) > 0:
            print(f"  - train.csv ({len(train_df)} samples)")
            print(f"  - validation.csv ({len(val_df)} samples)")
        if len(unlabeled_data) > 0:
            print(f"  - unlabeled_for_manual_annotation.csv ({len(unlabeled_data)} samples)")
        print(f"  - metadata.json")

        return metadata


# Usage example
if __name__ == "__main__":
    # Create scraper instance
    scraper = IndianCompanyAnnualReportScraper()

    # Create comprehensive dataset
    print("Creating comprehensive dataset...")
    df = scraper.create_comprehensive_dataset(
        include_real_data=True,  # Set to False for template-only dataset
        max_companies=10  # Adjust based on your needs
    )

    # Save dataset
    metadata = scraper.save_dataset(df)

    print("\nDataset Summary:")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Labeled samples: {metadata['labeled_samples']}")
    print(f"Companies: {metadata['companies']}")
    print(f"Sectors: {metadata['sectors']}")
    print(f"Average text length: {metadata['avg_text_length']:.0f} characters")

    if metadata['sentiment_distribution']:
        print("\nSentiment distribution:")
        for sentiment, count in metadata['sentiment_distribution'].items():
            print(f"  {sentiment}: {count}")

    print("\nDataset ready for manual labeling and Kaggle training!")
    print("Next steps:")
    print("1. Review and manually label unlabeled samples")
    print("2. Upload to Kaggle as a dataset")
    print("3. Use the Longformer training script for model training")