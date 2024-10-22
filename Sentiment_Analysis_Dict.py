'''This py file conatins dictionaries used in querying news articles'''

# Define a dictionary of queries for specific equity markets
'''
- S&P 500 [Average]: Represents the overall performance of the largest 500 companies in the U.S., serving as a benchmark for the U.S. equity market.
- NASDAQ [tech]: Focuses on technology and innovation-driven companies, often seen as a barometer for tech stocks.
- XLI [industrials]: Covers companies in the industrial sector, including manufacturing and transportation.
- XLF [financials]: Includes banks, investment funds, insurance companies, and real estate firms.
- XLRE [real estate]: Focuses on real estate investment trusts (REITs) and real estate companies.
- XLP [Consumer staples]: Covers essential products like food, beverages, and household items.
- XLY [Discretionary staples]: Represents non-essential goods and services, including retail and automotive sectors.
- XLV [healthcare]:  Includes pharmaceuticals, biotechnology, medical devices, and healthcare providers.
- NYMEX [energy]: Focuses on energy commodities, particularly oil and gas.
- XLB [materials]: This ETF provides exposure to companies in the materials sector, including those involved in mining, metals, chemicals, and forestry.
- XLC [communication services]: This ETF includes companies in telecommunications, media, and entertainment.
- XLU [utilities]: This ETF covers companies that provide essential services such as electricity, gas, and water.

'''
query_dict = {
    "S&P 500": 'equity OR stocks OR shares OR securities OR capital OR ownership OR investments OR market OR equities OR "publicly traded companies" OR "stock market" OR recession OR S&P OR CEO OR economy OR IPO OR invest OR AI OR GPU OR tax OR ROI OR politic OR election OR index OR fund',
    
    "NASDAQ": 'NASDAQ OR tech OR technology OR "publicly traded technology companies" OR innovation OR "software stocks" OR "hardware stocks" OR AI OR NVIDIA OR GPU OR Apple OR FAANG OR Facebook OR Meta OR Tesla OR quantum OR science OR space OR discovery OR Cobalt OR semiconductors',
    
    "XLI": 'industrials OR manufacturing OR "transportation stocks" OR "industrial sector" OR logistics OR supply chain OR machinery OR "construction stocks" OR "aerospace and defense"',
    
    "XLF": 'financials OR finance OR banks OR "investment banks" OR "financial services" OR insurance OR "asset management" OR "wealth management" OR "financial technology" OR "capital markets" OR Credit OR recession OR default OR gold ',
    
    "XLRE": 'real estate OR REITs OR "real estate investment trusts" OR "property management" OR "commercial real estate" OR "residential real estate" OR "real estate market"',
    
    "XLP": 'consumer OR "consumer staples" OR "essential products" OR "food and beverages" OR "household items" OR "grocery stores" OR "retail" OR "supermarkets"',
    
    "XLY": 'consumer OR "consumer discretionary" OR "retail stockas" OR "luxury goods" OR "automotive" OR "travel" OR "entertainment" OR "restaurants" OR "consumer services"',
    
    "XLV": 'healthcare OR pharmaceuticals OR biotechnology OR "medical devices" OR "healthcare providers" OR "health insurance" OR "clinical trials" OR "healthcare technology" OR cancer or vaccine or pfizier',
    
    "NYMEX": 'energy OR "oil prices" OR "natural gas" OR commodities OR "energy sector" OR "fossil fuels" OR "renewable energy" OR "energy market" OR crude OR brent OR refinery',
    
    "XLB": 'materials OR mining OR metals OR gold OR zinc OR platinum OR copper OR palladium OR chemicals OR forestry OR "construction materials" OR "raw materials" OR "commodity prices" OR semiconductors OR cobalt OR Dow OR Linde OR Newmont',
    
    "XLC": 'communication OR "telecommunications" OR media OR "social media" OR "broadcasting" OR "entertainment" OR "internet services" OR alphabet OR GOOGL OR META OR NFLX OR netflix or telegram or twitter',
    
    "XLU": 'utilities OR "electricity" OR "water services" OR "gas companies" OR "renewable utilities" OR "utility sector" OR "infrastructure" OR DUK OR nextera OR Southern Company',
}

# add a dictionary for equity markets that fall the different sectors
#there are 11 sectors get key words and equity markets for each



equity_markets_dict = {
    "S&P 500": {
        "etfs": ['SPY', 'IVV', 'VOO'],  # ETFs that track the S&P 500
        "etf_description": "Represents the overall performance of the largest 500 companies in the U.S., serving as a benchmark for the U.S. equity market."
    },
    "NASDAQ": {
        "etfs": ['QQQ'],  # ETF that tracks the NASDAQ-100
        "etf_description": "Focuses on technology and innovation-driven companies, often seen as a barometer for tech stocks."
    },
    "XLI": {
        "etfs": ['XLI'],  # Industrial Select Sector SPDR Fund
        "etf_description": "Covers companies in the industrial sector, including manufacturing and transportation."
    },
    "XLF": {
        "etfs": ['XLF'],  # Financial Select Sector SPDR Fund
        "etf_description": "Includes banks, investment funds, insurance companies, and real estate firms."
    },
    "XLRE": {
        "etfs": ['XLRE'],  # Real Estate Select Sector SPDR Fund
        "etf_description": "Focuses on real estate investment trusts (REITs) and real estate companies."
    },
    "XLP": {
        "etfs": ['XLP'],  # Consumer Staples Select Sector SPDR Fund
        "etf_description": "Covers essential products like food, beverages, and household items."
    },
    "XLY": {
        "etfs": ['XLY'],  # Consumer Discretionary Select Sector SPDR Fund
        "etf_description": "Represents non-essential goods and services, including retail and automotive sectors."
    },
    "XLV": {
        "etfs": ['XLV'],  # Health Care Select Sector SPDR Fund
        "etf_description": "Includes pharmaceuticals, biotechnology, medical devices, and healthcare providers."
    },
    "NYMEX": {
        "etfs": ['USO', 'UNG'],  # ETFs for oil and natural gas
        "etf_description": "Focuses on energy commodities, particularly oil and gas."
    },
    "XLB": {
        "etfs": ['XLB'],  # Materials Select Sector SPDR Fund
        "etf_description": "Provides exposure to companies in the materials sector, including those involved in mining, metals, and chemicals."
    },
    "XLC": {
        "etfs": ['XLC'],  # Communication Services Select Sector SPDR Fund
        "etf_description": "Includes companies in telecommunications, media, and entertainment."
    },
    "XLU": {
        "etfs": ['XLU'],  # Utilities Select Sector SPDR Fund
        "etf_description": "Covers companies that provide essential services such as electricity, gas, and water."
    },
}