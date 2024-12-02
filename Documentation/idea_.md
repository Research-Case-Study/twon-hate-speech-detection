
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



# Inference 
Its the main Agent chat inference


```Inference Pipeline
- Since now we have RAG in place with:  query_qdrant()
- and We have classify()
- System Prompt that tells the agent about whole scenario
- user_input The message of user.

system_prompt 

messages=[
        {   
            "role": "system", 
            "content": ":) "
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

```

Now lets understand the inference.
an LLM Inference without LLM:


```

user_input

llm_response = LLM.chat(userinput)
llm_response goes back to user.

In background:

"""
System prompt





```







