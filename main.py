
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, pairwise_distances

# Sample existing matrix (rows represent images, columns represent clothing types/colors)
existing_matrix = np.array([[0, 1, 0, 2, 0, 0],
                            [0, 0, 0, 3, 0, 1],
                            [0, 0, 4, 0, 5, 0],
                            [0, 0, 0, 1, 5, 0],
                            [0, 6, 0, 0, 0, 0]])

# Sample new line representing a single clothing item in a new image
new_image_row = np.array([0, 0, 0, 0, 0, 2])
# User's choice of similarity metric
print("Choose a similarity metric:")
print("1. Cosine Similarity")
print("2. Euclidean Distance")
print("3. Manhattan Distance")
choice = int(input("Enter the number of your choice: "))


# Function to calculate similarity based on user's choice
def calculate_similarity(existing_matrix, new_image_row, choice):
    if choice == 1:
        # Cosine Similarity
        similarity_scores = cosine_similarity([new_image_row], existing_matrix)
    elif choice == 2:
        # Euclidean Distance
        similarity_scores = 1 / (1 + euclidean_distances([new_image_row], existing_matrix))
    elif choice == 3:
        # Manhattan Distance
        similarity_scores = 1 / (1 + manhattan_distances([new_image_row], existing_matrix))
    else:
        print("Invalid choice. Using default: Cosine Similarity")
        similarity_scores = cosine_similarity([new_image_row], existing_matrix)
    return similarity_scores


# Calculate similarity based on user's choice
similarity_scores = calculate_similarity(existing_matrix, new_image_row, choice)

# Find the index of the most similar existing image
most_similar_index = np.argmax(similarity_scores)

# Get the most likely clothing type from the most similar existing image
most_similar_clothing_types = existing_matrix[most_similar_index]

# Find the complementary clothing type (not of the same type as the one in the new item)
new_item_clothing_type = np.nonzero(new_image_row)[0][0]  # Find the clothing type in the new item
complementary_clothing_type = np.argmax(most_similar_clothing_types)
while complementary_clothing_type == new_item_clothing_type:
    # Keep finding the complementary type until it's different from the new item type
    most_similar_clothing_types[complementary_clothing_type] = 0
    complementary_clothing_type = np.argmax(most_similar_clothing_types)

# Get the associated color for the complementary clothing type from the new item
associated_color = most_similar_clothing_types[complementary_clothing_type]

# Print the complementary clothing type and its associated color
print("Recommended Clothing Type:", complementary_clothing_type)
print("Associated Color:", associated_color)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, pairwise_distances

# Sample existing matrix (rows represent images, columns represent clothing types/colors)
existing_matrix = np.array([[0, 1, 0, 2, 0, 0],
                            [0, 0, 0, 3, 0, 0],
                            [0, 0, 4, 0, 5, 0],
                            [0, 6, 0, 0, 0, 0]])

# Sample new line representing a single clothing item in a new image
new_image_row = np.array([0, 0, 0, 0, 7, 0])  # Only one non-zero element representing the new item

# User's choice of similarity metric
print("Choose a similarity metric:")
print("1. Cosine Similarity")
print("2. Euclidean Distance")
print("3. Manhattan Distance")
choice = int(input("Enter the number of your choice: "))

# Function to calculate similarity based on user's choice
def calculate_similarity(existing_matrix, new_image_row, choice):
    if choice == 1:
        # Cosine Similarity
        similarity_scores = cosine_similarity([new_image_row], existing_matrix)
    elif choice == 2:
        # Euclidean Distance
        similarity_scores = 1 / (1 + euclidean_distances([new_image_row], existing_matrix))
    elif choice == 3:
        # Manhattan Distance
        similarity_scores = 1 / (1 + manhattan_distances([new_image_row], existing_matrix))
    else:
        print("Invalid choice. Using default: Cosine Similarity")
        similarity_scores = cosine_similarity([new_image_row], existing_matrix)
    return similarity_scores

# Calculate similarity based on user's choice
similarity_scores = calculate_similarity(existing_matrix, new_image_row, choice)

# Find the indices of the most similar existing images
most_similar_indices = np.argsort(similarity_scores[0])[::-1]

# Initialize a dictionary to store frequencies of each existing image
image_frequencies = {}

# Count the frequency of each existing image in the top similar indices
for idx in most_similar_indices:
    image_str = ' '.join(map(str, existing_matrix[idx]))
    if image_str not in image_frequencies:
        image_frequencies[image_str] = 0
    image_frequencies[image_str] += 1

# Find the existing image with the highest frequency
most_frequent_image_str = max(image_frequencies, key=image_frequencies.get)
most_frequent_image_idx = most_similar_indices[np.where([' '.join(map(str, existing_matrix[idx])) == most_frequent_image_str for idx in most_similar_indices])]

# Get the most likely clothing type from the most frequent existing image
most_similar_clothing_types = existing_matrix[most_frequent_image_idx]

# Find the complementary clothing type (not of the same type as the new item)
new_item_clothing_type = np.nonzero(new_image_row)[0][0]  # Find the clothing type in the new item
complementary_clothing_type = np.argmax(most_similar_clothing_types)
while complementary_clothing_type == new_item_clothing_type:
    # Keep finding the complementary type until it's different from the new item type
    most_similar_clothing_types[complementary_clothing_type] = 0
    complementary_clothing_type = np.argmax(most_similar_clothing_types)

# Get the associated color for the complementary clothing type from the new item
associated_color = new_image_row[complementary_clothing_type]

# Print the complementary clothing type and its associated color
print("Complementary Clothing Type:", complementary_clothing_type)
print("Associated Color:", associated_color)
"""