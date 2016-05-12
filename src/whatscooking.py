#!/usr/local/bin/python

from collections import Counter
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset
with open('../input/train.json', 'r') as fh:
    train = json.load(fh)
    
# Calculate the ingredient frequencies
ingredients = []
for recipe in train:
    for ingred in recipe['ingredients']:
        ingredients.append(ingred.lower())
        
# Write the results to csv
with open('../output/ingredient_counts.csv','w') as csvfile:
    fieldnames = ['Name','Count']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for key, value in Counter(ingredients).most_common():
        writer.writerow([key.encode(encoding='UTF-8',errors='ignore'), str(value)])

# Plot the top five ingredients (by count)
labels, values = zip(*Counter(ingredients).most_common(5))

indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.savefig('../output/top_five_ingredients_by_count.png')
plt.close()

# Calculate the cuisine frequencies
cuisines = []
for recipe in train:
	cuisines.append(recipe['cuisine'].lower())
	
# Write the results to csv
with open('../output/cuisine_counts.csv','w') as csvfile:
    fieldnames = ['Name','Count']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for key, value in Counter(cuisines).most_common():
        writer.writerow([key.encode(encoding='UTF-8',errors='ignore'), str(value)])

# Plot the top ten cuisines (by count)
labels, values = zip(*Counter(cuisines).most_common(5))

indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.savefig('../output/top_five_cuisines_by_count.png')
plt.close()
