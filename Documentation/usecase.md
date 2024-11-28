### Functional Inference Flow.
```Functional Flow.



user input

classified_output = Classifier ( userinput )
query_knowledge_graph_result = query_knowledge_graph(userinput ) 
rag_knowledge_base_result = RAG_knowledge_base(userinput ) # similarity search

output = LLM ( Prompt + classified_output + UserInput+ query_knowledge_graph_result, rag_knowledge_base_result ) ## rag_knowledge_base_result, and fact check is only added if their search similarity was high. 
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
                        +---------------------------+       +---------------------------+
                        | Query Knowledge Graph     | <---> | RAG: Knowledge Base       |
                        | - Relevant Data Search    |       | - Similarity Search       |
                        +---------------------------+       +---------------------------+
                                     |                               |
                                      +-------------------------------+
                                                      |
                                                      v
                   +---------------------------------------------------------+
                   |                PROMPT                                    |
                   |    Combine Outputs for final lLM input                   |
                   |  - Classifier Output                                   |
                   |  - Retrival Knowledge base (if similarity high)        |
                   |  - Knowledge Base -RAG (if similarity high)               |
                   |  - Knowledge Graph                |
                   +---------------------------------------------------------+
                                     |
                                     v
                          +------------------------------+
                          |  LLM Generates Output        |
                          |  (Based on all Inputs)       |
                          |   PROMPT                    |
                          |  - User Input + Classified  | 
                          |    Output + Knowledge graph |
                          |    Information + RAG Knowledge|
                          |    Base Results                 |
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


