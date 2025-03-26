# Hate Speech Detection Survey

## Objectives
- Identify the most effective prompting technique for explanation generation.
- Understand participant preferences on explanation clarity, plausibility, and contextual accuracy.
- Explore attitudes toward AI-based hate speech identification and moderation.
- Provide guidance for the design of transparent and trustworthy AI moderation systems.

---

## Repository Structure

### data/
- **raw/**
  - `Final Survey Results.csv`: Original survey responses directly exported from LimeSurvey.

- **processed/**
  - `LimeSurvey_data_pivoted.csv`: Pivoted version of raw data for simplified analysis.
  - `LimeSurvey_data_pivoted_sorted.csv`: Initial sorted dataset for analysis.
  - `LimeSurvey_data_pivoted_sorted_2.csv`: Intermediate refined sorting.
  - `LimeSurvey_data_pivoted_sorted_3.csv`: Final dataset prepared for detailed analysis and visualization.

### notebooks/
- `LimeSurvey_data_preperation.ipynb`: Data preprocessing and preparation steps.
- `Survey_Evaluation_Code.ipynb`: Initial exploratory data analysis.
- `Final_Evaluation_Code.ipynb`: Final analysis, visualizations, and interpretation of survey results.

### scripts/
- `EvaluationCodeTemplate.py`: Python script template for conducting analyses on survey data.

### explanation_selections/
- Individual and final documentation of tweet-explanation pair selections made by the research team.

### survey_logic/
- `SurveyStructureOverview.md`: Overview of survey structure, logic, and questions.
- `survey_logic_file.html`: Exported HTML file showing detailed survey implementation in LimeSurvey.

---

## Tools & Technologies
- **Programming Language:** Python
- **Libraries:** `pandas`, `matplotlib`, `re`
- **Survey Platform:** LimeSurvey (locally hosted)

