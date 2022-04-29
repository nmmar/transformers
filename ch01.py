from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

classifier(
    "I root for Blaugrana",
    candidate_labels=["sports", "culture", "politics"]
)

# Text generation
generator = pipeline("text-generation")
generator("In this course, we will teach you how to",\
    num_return_sequences=5, max_length=25)


generator_fin = pipeline('text-generation', model='Finnish-NLP/gpt2-finnish')
generator("Tekstiä tuottava tekoäly on", max_length=30, num_return_sequences=5)


# Mask filling
unmasker = pipeline("fill-mask")