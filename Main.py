# important import
from crewai import Agent, Task, Crew, Process, LLM
import agentops 
import os
from pydantic import BaseModel,Field
from typing import List
from tavily import TavilyClient
from crewai.tools import tool
from scrapegraph_py import Client
import json
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

# variable
no_keywords = 1
search_client = TavilyClient(api_key="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")
scrape_client = Client(api_key="sgai-ee4939b1-9e5d-4e00-bda6-d87d724dc2f2")
about_company = "Rankyx is a company that provides AI solutions to help websites refine their search and recommendation systems."
company_context = StringKnowledgeSource(content=about_company)
# API KEYS
os.environ["MISTRAL_API_KEY"] = "cJgIydUto15kD2r5J5ESuknlDXQ9WnjU"
os.environ["AGENTOPS_API_KEY"] = "9761238a-3851-43eb-976d-fa790cf86e3b"

# initialize  Agent ops
agentops.init (
    api_key ="9761238a-3851-43eb-976d-fa790cf86e3b",
    skip_auto_end_session=True
)

# make Directory

output_dir = './crew-ai'
os.makedirs(output_dir, exist_ok=True)

# Large Language Model "openai"

basicLLM  = LLM(
    model="mistral/mistral-large-latest",
    temperature=0.7
)
# output json

class Suggested_Search_Queries(BaseModel):
    queries: List[str] = Field(..., title="Suggested Search Queries", min_items=1, max_items=no_keywords)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////
# start create agent A

Search_Queries_Recommendation_Agent = Agent(
    role="Search Queries Recommendation Agent",
    goal="\n".join([
        "To provide a list of suggested search queries to be passed to the search engine.",
        "The queries must be varied and looking for specific items."
    ]),
    backstory="An expert in search optimization, generating precise and varied queries to enhance search accuracy.",
    llm=basicLLM,
    verbose=True
)


search_queries_recommendation_task = Task(
    description="\n".join(
        [
            "Rankyx is looking to buy {product_name} at the best prices (value for a price strategy)",
            "The company target any of these websites to buy from: {websites_list}",
            "The company wants to reach all available products on the internet to be compared later in another stage.",
            "The stores must sell the product in {country_name}",
            "Generate at maximum {no_keywords} queries.",
            "The search query must reach an ecommerce webpage for product, and not a blog or listing page.",
        ]
    ),
    expected_output="A JSON object containing a list of suggested search queries.",
    output_json=Suggested_Search_Queries,
    output_file=os.path.join(output_dir, "step_1_suggested_search_queries.json"),
    agent=Search_Queries_Recommendation_Agent,
)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////
# start create agent B

@tool 
def Search_Engine_Tool(query: str):
    """useful for search-based queries . use this to find current information about any query related pages using search engine"""
    return search_client.search(query)


class SignleSearchResult(BaseModel):
    title: str
    url: str = Field(..., title="the page url")
    content: str
    score: float
    search_query: str
class AllSearchResults(BaseModel):
    results: List[SignleSearchResult]


Search_Engine_Agent = Agent(
    role="Search Engine Agent",
    goal="To search for products based on the suggested search query",
    backstory="The agent is designed to help in looking for products by searching for products based on the suggested search queries.",
    llm=basicLLM,
    verbose=True,
    tools=[Search_Engine_Tool],
)

Search_Engine_Task = Task(
    description="\n".join(
        [
            "The task is to search for products based on the suggested search queries.",
            "You have to collect results from multiple search queries.",
            "Ignore any susbicious links or not an ecommerce single product website link.",
            "Ignore any search results with confidence score less than ({score_th}) .",
            "The search results will be used to compare prices of products from different websites.",
            "give me {no_keywords} products only",
        ]
    ),
    expected_output="A JSON object containing the search results.",
    output_json=AllSearchResults,
    output_file=os.path.join(output_dir, "step_2_search.json"),
    agent=Search_Engine_Agent,
)
# ///////////////////////////////////////////////////////////////////////////////////////////////////////
# start create agent C

# Specification products class
class ProductSpec(BaseModel):
    specification_name: str
    specification_value: str
# the class for each product
class SingleExtractedProduct(BaseModel):
    page_url: str = Field(..., title="The original url of the product page")
    product_title: str = Field(..., title="The title of the product")
    product_image_url: str = Field(..., title="The url of the product image")
    product_url: str = Field(..., title="The url of the product")
    product_current_price: float = Field(..., title="The current price of the product")
    product_original_price: float = Field(
        title="The original price of the product before discount. Set to None if no discount",
        default=None,
    )
    product_discount_percentage: float = Field(
        title="The discount percentage of the product. Set to None if no discount",
        default=None,
    )

    product_specs: List[ProductSpec] = Field(
        ...,
        title="The specifications of the product. Focus on the most important specs to compare.",
        min_items=1,
        max_items=5,
    )

    agent_recommendation_rank: int = Field(
        ...,
        title="The rank of the product to be considered in the final procurement report. (out of 5, Higher is Better) in the recommendation list ordering from the best to the worst",
    )
    agent_recommendation_notes: List[str] = Field(
        ...,
        title="A set of notes why would you recommend or not recommend this product to the company, compared to other products.",
    )

# all products
class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

#  declaraton tool
@tool
def web_scraping_tool(page_url: str):
    """
    An AI Tool to help an agent to scrape a web page

    Example:
    web_scraping_tool(
        page_url="https://www.noon.com/egypt-en/15-bar-fully-automatic-espresso-machine-1-8-l-1500"
    )
    """
    details = scrape_client.smartscraper(
        website_url=page_url,
        user_prompt="Extract ```json\n"
        + SingleExtractedProduct.schema_json()
        + "```\n From the web page",
    )
    
    return {"page_url": page_url, "details": details}

# Agent c
Scraping_Agent = Agent(
    role="Web scraping agent",
    goal="To extract details from any website",
    backstory="The agent is designed to help in looking for required values from any website url. These details will be used to decide which best product to buy.",
    llm=basicLLM,
    tools=[web_scraping_tool],
    verbose=True,
)
# Task of Agent c
Scraping_Task = Task(
    description="\n".join(
        [
            "The task is to extract product details from any ecommerce store page url.",
            "The task has to collect results from multiple pages urls.",
            "Collect the best {top_recommendations_no} products from the search results.",
        ]
    ),
    expected_output="A JSON object containing products details",
    output_json=AllExtractedProducts,
    output_file=os.path.join(output_dir, "step_3_search_results.json"),
    agent=Scraping_Agent,
)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

# start create Agent D
# Agent D
Procurement_Report_Author_Agent = Agent(
    role="Procurement Report Author Agent",
    goal="To generate a professional HTML page for the procurement report",
    backstory="The agent is designed to assist in generating a professional HTML page for the procurement report after looking into a list of products.",
    llm=basicLLM,
    verbose=True,
)
# Task for Agent D

Procurement_Report_Author_Task = Task(
    description="\n".join(
        [
            "The task is to generate a professional HTML page for the procurement report.",
            "You have to use Bootstrap CSS framework for a better UI.",
            "Use the provided context about the company to make a specialized report.",
            "The report will include the search results and prices of products from different websites.",
            "The report should be structured with the following sections:",
            "1. Executive Summary: A brief overview of the procurement process and key findings.",
            "2. Introduction: An introduction to the purpose and scope of the report.",
            "3. Methodology: A description of the methods used to gather and compare prices.",
            "4. Findings: Detailed comparison of prices from different websites, including tables and charts.",
            "5. Analysis: An analysis of the findings, highlighting any significant trends or observations.",
            "6. Recommendations: Suggestions for procurement based on the analysis.",
            "7. Conclusion: A summary of the report and final thoughts.",
            "8. Appendices: Any additional information, such as raw data or supplementary materials.",
        ]
    ),
    expected_output="A professional HTML page for the procurement report.",
    output_file=os.path.join(output_dir, "step_4_procurement_report.html"),
    agent=Procurement_Report_Author_Agent,
)


# ///////////////////////////////////////////////////////////////////////////////////////////////////////
# create CrewAI
Rankyx_crew = Crew(
    agents=[
        Search_Queries_Recommendation_Agent,
        Search_Engine_Agent,
        Scraping_Agent,
        Procurement_Report_Author_Agent,
    ],
    tasks=[
        search_queries_recommendation_task,
        Search_Engine_Task,
        Scraping_Task,
        Procurement_Report_Author_Task,
    ],
    process=Process.sequential,
    knowledge_sources=[company_context],
)
# run CrewAI
crew_results = Rankyx_crew.kickoff(
    inputs={
        "product_name": "table",
        "websites_list": ["www.amazon.eg", "www.jumia.com.eg", "www.noon.com/egypt-en"],
        "country_name": "Egypt",
        "no_keywords": 2,
        "score_th":0.1,
        "top_recommendations_no" : 2
    }
)
