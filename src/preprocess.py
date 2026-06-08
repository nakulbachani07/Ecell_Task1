import re # regular expression library: Find multiple spaces/newlines/tabs and replace them with one space.Regular expressions are used to find and replace text patterns.

TEXT_COLUMNS = [
    "Business" ,
    "Risk Factors",
    "Management’s Discussion and Analysis of Financial Condition and Results of Operations"
]

def combine_text_sections(row): #Keys of dictionary = column names #Values of dictionary = data in that row
    """
    Combines importantt 10-K sections into one text"""

    combined_text=""

    for col in TEXT_COLUMNS:
        value = row.get(col,"") #gets the value from that column.If it does not exist, give empty text instead.
        
        if value is not None:
            combined_text += " " + str(value)

    return combined_text

def clean_text(text):
    """
    Cleans raw filing text
    """

    if text is None:
        return ""
    
    text = str(text)

    #lowercase
    text = text.lower()

    #remove html - like tags
    text = re.sub(r"<.*?>" , " ", text) #re.sub(pattern, replacement, text)

    #remove urls
    text = re.sub(r"http\S+|www\S+", " ", text) # | - or , http\S+ → anything starting with http followed by non-space characters

    #remove unnecesary characters
    text = re.sub(r"[^a-zA-Z0-9\s.,;:%$-]", " ", text) #The ^ inside square brackets means:anything NOT in this list , keep ^a-zA-Z0-9\s.,;:%$- , remove additional

    #remove extra spaces
    text = re.sub(r"\s+"," ", text).strip() #.strip() - Removes spaces from beginning and end.

    return text

def truncate_text(text, max_chars=10000):
    """
    Limits text length to avoid memory/trainig issues"""

    return text[:max_chars]


  

    
                                 