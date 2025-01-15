# Classify the label without model seeing the actual label, untrained model.
prompt_1 = """
Lets classify the following text for hate speech.

if its hatespeech your reply will be: "hate speech"
if its not hatespeech your reply will be : "normal"
if its offensive and not hatespeech, your reply will be : "offensive"

Here is user input and reply with only one word ONLY
"""


# Classify the label with model seeing the actual label, untrained model. (Aman's prompt)
prompt_2 = """
you are AI agent which is specialized in analyzing a text for 3 different classes which is hate speech, offensive and normal.

available data: 

{'post_id': ## post id . ignore it",
    " 'tweet_text': #its the main user text that you need to classify.",
    " 'key_features': [] # list of the words that made a decision, important feature",
    " 'target': # targetted audience. ",
    " 'label': # either offensive, hate speech or normal. completely ignore this while predicting your own label",
    " }
notes:
1) Your output should only contain the predicted label based on the tweet text provided without any explanation or reasoning.
2) while predicting the label you should also consider the possible context for the text which the user might have in mind when writing the tweet
"""