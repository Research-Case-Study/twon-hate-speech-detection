

### Usecase
``` Usecase


                          +---------------------------+
                          |      Start                |
                          +---------------------------+
                                     |
                                     v
                          +---------------------------+
                          |   Detect Text Input       |
                          |  (e.g., message, comment) |
                          +---------------------------+
                                     |
                                     v
                          +---------------------------+
                          |  Classify Text            |
                          |  - Basic Classification   |
                          |  - Multi-Class            |
                          +---------------------------+
                                     |
                                     v
                          +---------------------------+
                          |  Analyze Text Context     |
                          |  (Conversation History)   |
                          +---------------------------+
                                     |
                                     v
            +-------------------------------------------+
            |   Is Text Hate Speech or Offensive?      |
            +-------------------------------------------+
                   |                             |
       +-----------+-----------+        +--------+--------+
       |                       |        |                 |
  +------------+         +------------+   +-----------------+
  |  Implicit   |         |  Explicit  |   |  Sarcasm or     |
  | Hate Speech |         |  Hate     |   |  Humor Detection|
  | Recognition |         | Detection |   +-----------------+
  +------------+         +------------+          |
       |                        |                |
       v                        v                v
+-------------------+   +-------------------+  +---------------------+
| Assign Severity   |   | Flag as Hate Speech|  | Flag as Offense     |
| Level (Intensity) |   |                    |  |                     |
+-------------------+   +-------------------+  +---------------------+
       |                        |                      |
       v                        v                      v
+------------------+   +------------------+    +----------------------+
| Provide Feedback |   | Provide Feedback |    | Provide Feedback      |
| (to User/ Admin) |   | (to User/ Admin) |    | (to User/ Admin)      |
+------------------+   +------------------+    +----------------------+
       |                        |                      |
       +------------------------+----------------------+
                            |
                            v
              +------------------------------+
              |  Offer Moderation Suggestions |
              +------------------------------+
                            |
                            v
              +-----------------------------+
              |   Notify Users of Violations |
              +-----------------------------+
                            |
                            v
                +----------------------------+
                |    End Process / Loop Back |
                +----------------------------+


```


### Functional Inference Flow.
```Functional Flow.



user input

classified_output = Classifier ( userinput )
rag_fact_check_result = RAG_fact_check(userinput ) # similarity search
rag_knowledge_base_result = RAG_fact_check(userinput ) # similarity search

output = LLM ( Prompt + classified_output + UserInput+ rag_fact_check_result, rag_knowledge_base_result ) ## rag_knowledge_base_result, and fact check is only added if their search similarity was high. 
## classified_output just helps perception of model, and it may not misunderstand the intention of user.


                          +---------------------------+
                          |      Start                |
                          +---------------------------+
                                     |
                                     v
                          +---------------------------+
                          |   Detect User Input       |
                          |  (Message, Comment, etc.) |
                          +---------------------------+
                                     |
                                     v
                          +---------------------------+
                          |  Classify Text            |
                          |  - Basic Classification   |
                          |  - Multi-Class            |
                          +---------------------------+
                                     |
                                     v
                          +---------------------------+
                          |  RAG: Perform Fact Check  |
                          |  - Similarity Search for  |
                          |    Fact-Check Information |
                          +---------------------------+
                                     |
                                     v
                          +---------------------------+
                          |  RAG: Knowledge Base      |
                          |  - Similarity Search for  |
                          |    Knowledge Base Content |
                          +---------------------------+
                                     |
                                     v
                   +--------------------------------------------+
                   |   Combine Outputs for Final Response     |
                   |  - Classifier Output                      |
                   |  - Fact Check (if similarity high)        |
                   |  - Knowledge Base (if similarity high)    |
                   +--------------------------------------------+
                                     |
                                     v
                          +---------------------------+
                          |  LLM Generates Output     |
                          |  (Based on all Inputs)   |
                          |   PROMPT                  |
                          |  - User Input + Classified|
                          |    Output + Fact-Checked  |
                          |    Information + Knowledge|
                          |    Base Results           |
                          +---------------------------+
                                     |
                                     v
      +--------------------------------------------------------------------------------------+
      | User Receives Output Generatic answer of an LLM that was well informed of the intention |
      +---------------------------------------------------------------------------------------+
                                     |
                                     v
                          +---------------------------+
                          |    End Process            |
                          +---------------------------+

```


