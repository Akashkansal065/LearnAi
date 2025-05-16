
from bs4 import BeautifulSoup
import json

# Load the HTML content from a file
with open('/Users/akash.kansal/Documents/GitHub/LearnAii/02/temp.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Function to generate XPath for an element


def get_xpath(element):
    components = []
    for parent in element.parents:
        siblings = parent.find_all(element.name, recursive=False)
        if len(siblings) == 1:
            components.append(element.name)
        else:
            index = siblings.index(element) + 1
            components.append(f'{element.name}[{index}]')
        element = parent
    components.reverse()
    return '/' + '/'.join(components)

# Function to get all possible locators for an element


def get_locators(element):
    return {
        'tag': element.name,
        'xpath': get_xpath(element),
        'css_selector': element.name,
        'id': element.get('id'),
        'class': element.get('class'),
        'name': element.get('name'),
        'attributes': element.attrs
    }


# Extract all elements and their locators
elements = soup.find_all()
locators_list = [get_locators(element) for element in elements]

# Save or print the results
with open('locator_output.json', 'w', encoding='utf-8') as f:
    json.dump(locators_list, f, indent=2, ensure_ascii=False)

print("Locator extraction complete. Results saved to 'locator_output.json'.")
