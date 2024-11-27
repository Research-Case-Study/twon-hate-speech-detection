



### Initial Dataset in mind which is a Multilabel Dataset.

| Tweet Text                            | Religion | Racism | Politics |
|---------------------------------------|----------|--------|----------|
| "No hate in this tweet"               | 0        | 0      | 0        |
| "Hate speech on religion"             | 1        | 0      | 0        |
| "Discriminatory remarks on race"      | 0        | 1      | 0        |
| "Hateful political comments"          | 0        | 0      | 1        |
| "Religion and politics are targets"   | 1        | 0      | 1        |
| "Racism and politics in hate speech"  | 0        | 1      | 1        |
| "Hate related to religion and racism" | 1        | 1      | 0        |
| "Hate about all topics"               | 1        | 1      | 1        |
| "Casual political remark"             | 0        | 0      | 1        |
| "Religious discrimination post"       | 1        | 0      | 0        |
| "No racism or hate"                   | 0        | 0      | 0        |
| "Targeting race and politics"         | 0        | 1      | 1        |
| "Religious hate speech online"        | 1        | 0      | 0        |
| "Offensive race-based tweet"          | 0        | 1      | 0        |
| "Tweet about all three issues"        | 1        | 1      | 1        |




###  However Depending on the dataset available, we go with a good quality dataset.

### Multiclass Dataset.  TopicID , HateLabel are the main table here which are both Multiclass

Source dataset: https://github.com/avaapm/hatespeech

The dataset contains 

| TweetID           | LangID | TopicID   | Label_1  | Label_2  | Label_3  | Label_4  | Label_5  | HateLabel |
|--------------------|--------|-----------|----------|----------|----------|----------|----------|-----------|
| 1344792819660189696 | 1      | 0 (Religion) | Normal   | Offensive | Hate     | Normal   | Normal   | Offensive |
| 1344767690225954816 | 1      | 0 (Religion) | Normal   | Normal   | Normal   | Offensive | Normal   | Normal    |
| 1344762665764343809 | 1      | 0 (Religion) | Normal   | Normal   | Offensive | Normal   | Normal   | Offensive |
| 1344760030537670657 | 1      | 0 (Religion) | Offensive | Normal   | Normal   | Normal   | Hate     | Hate      |
| 1344737275872235520 | 1      | 0 (Religion) | Normal   | Normal   | Normal   | Normal   | Normal   | Normal    |
| 1344736054188924931 | 1      | 0 (Religion) | Normal   | Normal   | Normal   | Normal   | Normal   | Normal    |
| 1344717408288825344 | 1      | 0 (Religion) | Normal   | Normal   | Normal   | Normal   | Normal   | Normal    |
| 1344702716900216834 | 1      | 1 (Gender)   | Hate     | Hate     | Normal   | Normal   | Offensive | Hate      |
| 1344701188231540736 | 1      | 1 (Gender)   | Normal   | Normal   | Normal   | Normal   | Normal   | Normal    |
| 1344694354762469377 | 1      | 1 (Gender)   | Normal   | Normal   | Normal   | Normal   | Normal   | Normal    |
| 1344693703487664130 | 1      | 2 (Race)     | Normal   | Normal   | Normal   | Normal   | Normal   | Normal    |
| 1344692706182897664 | 1      | 1 (Gender)   | Offensive | Offensive | Normal   | Normal   | Normal   | Offensive |
| 1344692359641133056 | 1      | 1 (Gender)   | Normal   | Normal   | Normal   | Normal   | Normal   | Normal    |
| 1344683235775803395 | 1      | 1 (Gender)   | Hate     | Normal   | Normal   | Normal   | Normal   | Hate      |
| 1344681665919115264 | 1      | 1 (Gender)   | Normal   | Normal   | Normal   | Normal   | Normal   | Normal    |

