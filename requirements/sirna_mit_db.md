sirna_mit_db Requirements Document

1. Overview

The goal of the sirna_mit_db Python script is to automate the extraction and enrichment of data from the MIT siRNA webpage. The script will perform the following tasks:
	•	Scrape a main table from https://web.mit.edu/sirna/sirnas-human.html.
	•	Extract and parse the table with headers:
	•	Target Gene
	•	siRNA ID#
	•	siRNA
	•	shRNA
	•	Human
	•	NCBI Probe #
	•	mRNA knockdown
	•	Protein knockdown
	•	For each URL found in the siRNA ID# column:
	•	Visit the linked page.
	•	Capture detailed information from the page:
	•	Target gene
	•	Sense sequence
	•	NCBI Database ID#
	•	Anti-sense sequence
	•	Append the extracted details as new columns to the original table.
	•	Calculate and return a numeric average value for one of the knockdown values (either “mRNA knockdown” or “Protein knockdown”).

2. Functional Requirements

2.1 Main Page Data Extraction
	•	Retrieve the main page:
	•	Use the URL: https://web.mit.edu/sirna/sirnas-human.html.
	•	Validate that the HTTP response status is 200 (OK).
	•	Parse the table:
	•	Locate the table that contains the specified headers.
	•	Extract each row of the table and store the data in an appropriate data structure (e.g., a pandas DataFrame).
	•	Ensure that the following columns are extracted:
	•	Target Gene
	•	siRNA ID#
	•	siRNA
	•	shRNA
	•	Human
	•	NCBI Probe #
	•	mRNA knockdown
	•	Protein knockdown

2.2 Detail Page Data Extraction
	•	Identify URLs:
	•	For each row, extract the URL from the siRNA ID# column.
	•	Handle relative and absolute URLs appropriately.
	•	Scrape individual detail pages:
	•	For each URL, send an HTTP request and verify that the page is accessible.
	•	Extract the following information from each detail page:
	•	Target gene
	•	Sense sequence
	•	NCBI Database ID#
	•	Anti-sense sequence
	•	Handle cases where a detail page may be missing one or more fields by logging a warning and continuing the process.

2.3 Data Aggregation
	•	Merge data:
	•	Enrich the original main table by adding new columns for each of the following fields (from the detail pages):
	•	Target gene (detailed)
	•	Sense sequence
	•	NCBI Database ID#
	•	Anti-sense sequence
	•	Compute averages:
	•	Choose one of the knockdown fields (either “mRNA knockdown” or “Protein knockdown”) and compute the numeric average of the values across the table.
	•	Validate that the knockdown values are numeric and handle any parsing errors gracefully.

2.4 Output
	•	Final data:
	•	Return or display the updated table that now includes the original columns along with the added detail columns.
	•	Output the computed average knockdown value (as a numeric result).
	•	Optionally, export the enriched table to a CSV file for further analysis.

3. Non-functional Requirements
	•	Language and Environment:
	•	The script must be implemented in Python 3.x.
	•	The code should be compatible with common operating systems (Linux, macOS, Windows).
	•	Dependencies:
	•	Use standard libraries and popular third-party packages, such as:
	•	requests (for HTTP requests)
	•	BeautifulSoup from bs4 (for HTML parsing)
	•	pandas (for data manipulation)
	•	Error Handling:
	•	Gracefully handle HTTP errors, connection timeouts, or malformed HTML.
	•	Log issues with sufficient detail to facilitate debugging (e.g., missing fields, URL failures).
	•	Ensure that if one detail page fails, the script continues processing the remaining pages.
	•	Code Quality:
	•	Code must be modular, separating concerns into functions or classes:
	•	Main page scraping and table extraction.
	•	Detail page scraping.
	•	Data aggregation and processing.
	•	Computation of averages.
	•	The script should include inline documentation and adhere to Python best practices (e.g., PEP 8).
	•	Unit tests should be implemented to cover key functions, especially for parsing and aggregation logic.
	•	Performance:
	•	Optimize for a reasonable response time when processing multiple detail pages.
	•	Consider implementing rate limiting or delays between HTTP requests if necessary to avoid overwhelming the server.

4. Detailed Implementation Outline

4.1 Module: Main Page Scraper
	•	Functionality:
	•	Fetch the main page content.
	•	Parse the HTML to locate the table containing the expected headers.
	•	Convert the table into a pandas DataFrame.
	•	Validation:
	•	Confirm that the expected headers exist in the table.
	•	Log and handle discrepancies.

4.2 Module: Detail Page Scraper
	•	Functionality:
	•	Iterate through the URLs extracted from the siRNA ID# column.
	•	For each URL:
	•	Fetch the page content.
	•	Parse the HTML to extract the four key fields:
	•	Target gene
	•	Sense sequence
	•	NCBI Database ID#
	•	Anti-sense sequence
	•	Return a mapping of detail data corresponding to each URL.
	•	Validation:
	•	Ensure that each extracted value is non-empty and log missing or unexpected data.

4.3 Module: Data Aggregation
	•	Functionality:
	•	Merge the detail page data with the main table.
	•	Ensure that rows align correctly (using the URL or a unique identifier).
	•	Validation:
	•	Check for consistency between the main table and the detail data.

4.4 Module: Computation of Knockdown Averages
	•	Functionality:
	•	Select one of the knockdown columns (e.g., mRNA knockdown).
	•	Convert the column data to a numeric type, handling any formatting issues.
	•	Calculate the average and log the result.
	•	Validation:
	•	Ensure that all values are numeric or are properly converted.
	•	Handle any conversion errors gracefully.

4.5 Module: Output Handling
	•	Functionality:
	•	Display or return the enriched DataFrame.
	•	Print or log the computed average knockdown value.
	•	Optionally, export the final DataFrame to a CSV file.

5. Testing and Validation
	•	Unit Testing:
	•	Write tests for the main page scraping function to ensure the table is correctly parsed.
	•	Write tests for the detail page parsing function with sample HTML content.
	•	Write tests for the data aggregation process.
	•	Write tests for numeric conversion and average calculation.
	•	Integration Testing:
	•	Run the complete script on a subset of data to verify the overall process.
	•	Test with edge cases, such as missing or malformed HTML content in detail pages.

6. Assumptions and Considerations
	•	The main page’s HTML structure is stable and includes the table with the specified headers.
	•	The URLs in the siRNA ID# column are valid and accessible.
	•	Knockdown values are numeric and in a format that can be easily converted (e.g., percentages, decimals).
	•	There may be slight variations in HTML markup across detail pages; the parser should be robust enough to handle minor inconsistencies.
	•	Respectful scraping: the script should be designed to avoid overwhelming the server (e.g., by implementing delays between requests if needed).

7. Future Enhancements
	•	Add command-line arguments for user customization (e.g., selecting which knockdown column to average).
	•	Implement caching to reduce repeated HTTP requests during development.
	•	Enhance logging with different levels (INFO, DEBUG, ERROR) and possibly output to a log file.
	•	Create a configuration file for adjustable parameters (e.g., URL, delay times, output file path).