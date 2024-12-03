
# Agent PREPERATION PIPELINE:

## 1 Dataset Preperation
``` 
Dataset preparation for Classifier and RAG:

multiclass or multilabel data ( whichever you guys finalize) 


1. For classifier : dataset-> X_train -> embeddings -> classifier...  | Y_train, remains as it is.
Dataset
   |
   v
+--------------------+
| Split into        |
| X_train and       |
| Y_train           |
+--------------------+
   |
   v
X_train -------------> Embeddings -------------> Classifier training
                                             

2. For RAG: dataset -> Use LLM to write EXPLAINATIONS .. why and how can this message be hate speech ( if Y_train was TRUE/ if that row in dataset was hate ) 
similarly explaiantion, if target label was False ( not hatespeech ) ( you guys finalize this specific line, but agent can go ahead without this one too ) 

Dataset
   |
   v
+--------------------+
| Use LLM to        |
| generate          |
| explanations      |
+--------------------+
   |
   v
If Y_train is TRUE: -----------------> Explain why/how this message could be hate speech.
If Y_train is FALSE: -----------------> Explain why/how this message is not hate speech.

-------------------------------------------------------------------
Now we have a CLASSIFIER, and EXPLAINATIONs

If dataset was true. accurate. then explainations are true too, and not subject to hallucinations since 

Final Output:
   - CLASSIFIER (Predictive Model) (For accurately detecting hate speech, for better accuracy, FOR TRIGGING HATE SPEECH context in the response WHEN USER does it)
   - EXPLANATIONS (Justifications for Predictions) ( FOR RAG ) (Can be used for LLM training but we are not going for that. just sharing info)

Key Considerations:
   - Accuracy of dataset ensures explanations are truthful and not hallucinated.
   - Difficulty: EASY (Dataset and tools are well-known, implementation is straightforward.)
   - Time Estimate: A few days to 1 week.


```
## RAG
Now that we have 2 datasets. now Lets focus on RAG preparation.

```

Vector DB: I chose Qdrant out of ( Qdrant, Chrome DB, Weaveate, pinecone, elastic search)
reason for Qdrant: Metadata. I can use it to do embeddings under two categories. hate or not hate. 
Vector db search query returns best matching contextually relevant info and similar looking words too, and puts a score on how relevant the info was.

Example:
If input query:  not about hate
RAG output:
[
    {
	"score": "0.92",
	"is_hate": False,
	"explaination" : "blah blah. here there will be some explainations that we discussed early" 
    },
    {
	"score": "0.60",
	"is_hate": True,
	"explaination" : "blah blah. here there will be some explainations that we discussed early" 
    }
]
. and If we also have (NOT HATE SPEECH) explanations too, the score can help us identity.
for example. if we didnt have Not Hate speech example. we would soly rely on score LIMITING ( if score > 0.7 , and if "is_hate" == classifier, only then include explainations " 

Example:
If input query:  not about hate
RAG output:
[
    {
	"score": "0.60",
	"is_hate": True,
	"explaination" : "blah blah. here there will be some explainations that we discussed early" 
    }
]

------
FLOW.

So we had explainations. 

we would create payload like this:

[
    {
	
	"is_hate": True,
	"type_of_hate" : "Religion"
	"explaination" : "blah blah. " 
    }
]

This will then be fed to qdrant vector db. 



And we will have a function:

payload = query_qdrant(user_message, k_top, classifier_output)

payload will be 
{
	"score": 0.9
	"is_hate": True,
	"type_of_hate" : "Religion"
	"explaination" : "blah blah. " 
    }



Time required : 1 day. I have been working with RAGs since 2023, March when pinecone, weaviate, chroma where being developed. And I have the code too.

```

RAG indexes
```
We will have one main INDEX for all the explainations which are fetched, based on userinput and classifier's decision.
And we will have multiple other index for each User. Each user will have its own index, so we can fetch his own specific history of chat to find if user talked about something in past.

THIS MEMORY THING is not the main focus of the project but it wont take any extra effort.


for example:
Main index: for all explainations, is_hate, type_of_hate.
user1_index: all chat history of user1
user2_index: all chat history of user2
user3_index: all chat history of user3
++++++++
user*_index: all chat history of user*


                   +----------------------------+
                   |      Vector Database       |
                   |----------------------------|
                   |                            |
                   |   +-------------------+    |
                   |   |   Main INDEX      |    |
                   |   |  *************    |    |
                   |   |  * Explanations * |    |
                   |   |  * is_hate      * |    |
                   |   |  * type_of_hate * |    |
                   |   |  *************    |    |
                   |   +-------------------+    |
                   |                            |
                   |  +---------+   +---------+ |
                   |  |  ****   |   |  ****   | |
                   |  |   **    |   |   **    | |
                   |  | User1   |   | User2   | |
                   |  | Index   |   | Index   | |
                   |  +---------+   +---------+ |
                   |                            |
                   |      +---------+           |
                   |      |  ****   |           |
                   |      |   **    |           |
                   |      | User3   |           |
                   |      | Index   |           |
                   |      +---------+           |
                   +----------------------------+

```



# AGENT Inference  
Its the main Agent chat inference


```Inference Pipeline
Since now we have
- query_qdrant(): RAG in place 
- classify(): To accurately classify the class.
- System Prompt: that tells the agent about whole scenario, what its supposed to do.  for example. we tell model to priortize the classifier's output. so user cannot convince it otherwise.
- user_input The message of user.


                                            user_input 
                                                 |
                                                 v
                               +----------------------------------------+
                               | Generate Embeddings                    |
                               | using STELLA or Nvidia's NV EMBED model|
                               +----------------------------------------+
                                       |                           |
                                       |                           v
                                       |           +----------------------------+
                                       |           | Classifier:               |
                                       |           | - Fetch correct class     |
                                       |           | - Identify type of hate   |
                                       |           +----------------------------+
                                       |                  |
                                       v                  v
                               +----------------------------+
                               | RAG Part:                 |
                               | - Use same embeddings     |
                               | - Perform cosine similarity|
                               | - Get "type_of_hate"      |
                               | - Get "is_hate" (bool)    |
                               | - Get "explanation"       |
                               | - Filter RAG results by   |
                               |   matching type_of_hate   |
                               | - Filter RAG for past chat|
                               |   history (memory)        |
                               | - Use score to add        |
                               |   confidence              |
                               +----------------------------+







                               Model Input Prompt:
                               +----------------------------------------------------------+
                               | System Prompt                                            |
                               | Messages History + Rag index's similar old messages                                         |
                               | User Input                                               |
                               | Classifier's Output (DECISION)                           |
                               | Explanations fetched from RAG                            |
                               +----------------------------------------------------------+
                                  |
                                  v
                               +----------------------------+
                               | LLM Processes Input Prompt |
                               +----------------------------+
                                  |
                                  v
                               +-------------------+
                               | Output Message to |
                               | User              |
                               +-------------------+
Experiments we can perform
# score differences. confidence.
#
#

```
### Python Example of flow

```
System Prompt in OpenAI payload style. 


system_prompt = "You are an Agent supposed to chat like ETC.. give priority to classifier's output. chat normally unless classifier detects hate speech"
user_input # comes from user
X_embeddings = SELLA_MODEL.encode(user_input)
classified_class = classifier(X_embeddings)
explaination = query_qdrant(X_embeddings, k_top=3, classified_class)  # logic will be inside the function.

context_for_assistant = "Here is the classified class fetched: "+  classified_class + "Explaination: " + explaination ######### IMPORTANT
context_for_assistant = context_for_assistant + RAG fetched user specific message history (memory)
payload = {
model: etc
messages=[
        {   
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_input
        }
        {
            "role": "system",
            "content": context_for_assistant
        }
    ]
}
response = openai.chat.completions.create(payload)
RESPONSE = response.choices[0].message
return RESPONSE

```

### Python example of flow RAW Transformers, no wrapper, no langchain, no openai

```

system_prompt = "You are an Agent supposed to chat like ETC.. give priority to classifier's output. chat normally unless classifier detects hate speech"
user_input # comes from user
X_embeddings = SELLA_MODEL.encode(user_input)
classified_class = classifier(X_embeddings)
explaination = query_qdrant(X_embeddings, k_top=3, classified_class)  # logic will be inside the function.




tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

X_embeddings = SELLA_MODEL.encode(user_input)
classified_class = classifier(X_embeddings)

context_for_assistant = "Here is the classified class fetched: "+  classified_class + "Explaination: " + explaination ######### IMPORTANT
context_for_assistant = context_for_assistant + RAG fetched user specific message history (memory)
input_ids = tokenizer(user_input , return_tensors="pt").input_ids

(IMPORTANT)
chat_payload = f""" {system_prompt} OR : "You are an Agent supposed to chat like ETC.. give priority to classifier's output. chat normally unless classifier detects hate speech"

(userchat history. such as) :

Human: Hi man, how are you doing.
AI:<s>Im good fella, how about you</s>
Human: what fella? how dare you call it. you Australian ***
context: {context_for_assistant}
AI:"""

output = model.generate(chat_payload, max_length=2000, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

Response = generated_text


<s> start of chat token
</s> end of statement token. we call it stop sequence. it is where the AI prediction streaming is stopped.


```




```

### Improvements
- We can improve classifier. As long as classifier is quite accurate, rest of the flow wont make mistake with identifying. Because this is working a main switch.
- Explanations 


I add more later



