import tensorflow_probability as tfp
from transformers import TapasTokenizer, TFTapasForQuestionAnswering
import pandas as pd

# Load the tokenizer and model
tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
model = TFTapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wtq')

# Create a sample table
data = {
    "Actors": ["Brad Pitt", "Leonardo DiCaprio", "George Clooney"],
    "Age": ["56", "45", "59"],
    "Number of Movies": ["87", "53", "69"]
}
table = pd.DataFrame.from_dict(data)

# Ask questions
queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

# Tokenize the input data
inputs = tokenizer(table=table, queries=queries, padding='max_length', return_tensors="tf")

# Pass the inputs to the model
outputs = model(**inputs)

# Extract the logits
logits = outputs.logits

# Process the logits to get the answers
answers = []
for query_index in range(len(queries)):
    # Get the predicted answer coordinates
    predicted_answer_coordinates = tokenizer.convert_logits_to_predictions(
        inputs, logits[query_index:query_index+1]
    )
    
    # Extract the answer from the table using the predicted coordinates
    answer_coordinates = predicted_answer_coordinates[0]
    answer_text = []
    for coord in answer_coordinates:
        if coord is not None:
            answer_text.append(table.iat[coord])
    answers.append(" ".join(answer_text))

# Print the answers
for query, answer in zip(queries, answers):
    print(f"Query: {query}")
    print(f"Answer: {answer}")