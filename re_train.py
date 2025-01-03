import re

text = "apple123banana456cherry"
result = re.split(r'\d+', text)
print("Split text:", result)  # Output: Split text: ['apple', 'banana', 'cherry']