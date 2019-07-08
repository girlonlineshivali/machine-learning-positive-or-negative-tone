from sklearn import tree

from sklearn.feature_extraction.text import CountVectorizer

print('')
print("Welcome to my first machine learning program! I am going to check whether your sentences in the code below are positive or negative. You can edit the sentences below and see how accurate I am.")
print('')
#dictionaries 

positive_texts = [
    "i love pancakes",
    "you love chicken",
    "she is good at design",
    "he is good at coding",
    "we are good at soccer",
    "she is beautiful",
    "she is amazing at drawing",

]

negative_texts =  [
    "we hate her", 
    "they hate us for winning",
    "you are bad at drawing",
    "he is bad at sleeping",
    "we hate doing chores",
    "he is awful at listening",
    "she is bad at math",
    "her singing is terrible",

]

test_texts = [
    "they love eating good food",
    "they are good at sleeping",
    "why do you hate doing chores",
    "they are almost always good at coding",
    "we are very bad at soccer",
    "she has a negative attitude",
    "he is stupid",
]

#training

training_texts = negative_texts + positive_texts
training_labels = ["negative"] * len(negative_texts) + ["positive"] * len(positive_texts)

vectorizer = CountVectorizer()
vectorizer.fit(training_texts)
print('')
print(vectorizer.vocabulary_)

training_vectors = vectorizer.transform(training_texts)
testing_vectors = vectorizer.transform(test_texts)

classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)
predictions = classifier.predict(testing_vectors)
print(predictions)

tree.export_graphviz(
    classifier,
    out_file='tree.dot',
    feature_names=vectorizer.get_feature_names(),
) 

def manual_classify(text):
    if "hate" in text:
        return "negative"
    if "bad" in text:
        return "negative"
    if "awful" in text:
        return "negative"
    if "negative" in text:
        return "negative"
    if "terrible" in text:
        return "negative"
    if "stupid" in text:
        return "negative"
    return "positive"

print('')
print("accuracy! negative only if you include hate, bad, awful, negative, stupid, or terrible.")

predictions = []
for text in test_texts:
    prediction = manual_classify(text)
    predictions.append(prediction)
print(predictions)

print('')
print("keep changing or adding sentences to see how well I can identify the tone.")
