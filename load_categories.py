import json

def load_categories(filename):
    """Load categories from a file containing json data

    Parameters:
    filename (string): the name of the file

    Returns:
    object:the categories object
   """
    
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name