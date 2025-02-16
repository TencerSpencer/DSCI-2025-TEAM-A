# DSCI-2025-TEAM-A
## Message Traffic Summarization
Team members: Christian Reza, Christian Young, Spencer Thomas

---
# Meeting Minutes

## 5 Feb 2025

**How do we do EDA work on data that we generate ourselves?**
 * For the assignment, we can use other datasets just for the assignments and present that.

**Need paper we are studying related to our project and link in MS Teams (Each person needs a unique paper).**
* Message summarization techniques (not too heavy though.)
  * Create one or two slides based on the paper we are reading.
* Some topics we may want to look for are about the models we can use and if we are able to add layers to the models.
  * Look into how we can add layers to these models as well.

**Do we need the messages at all?**
* Would it be easier to just generate data for the tables and use that? Probably, since we are not building message validation model, just a summarization one.
* Shouldn't have to worry about processing rules for the data (message sequences.)
  * Models understanding an API schema rather than a table schema (this may be big picture, not for the semester, though.)
* **Skip messages, instead create a schema for the data**

**Model Mayhem**
* What category of model do can we use?
  * Question/Answer model or Table Question Answering Model (Table Q&A model)
    * How do we create data to train these kinds of models.
    * We want a model that is interactive with the user.
  * Table Q&A Models can query a table and get the results, so we need a schema and not messages.
  * Q&A Model do not need a database schema to be able to get answers.
  * The type of model we use will most likely affect the kind of schema we use.
* **We don't have a dataset, creating one will be very time consuming. Are there any Q&A datasets available that we can use instead?**
  * Dig around Hugging Face datasets
    * https://huggingface.co/datasets?task_categories=task_categories:table-question-answering&sort=trending
    * https://huggingface.co/datasets?task_categories=task_categories:question-answering&sort=trending
    * Text-to-SQL model
      * Are there pre-trained models for that, if so, can we take two models and merge them together to create what we want? (That would be nice, would need do transfer learning to merge them).
        * Transfer learning could require a new dataset and PyTorch to add to the last layer.
* Start with the the most simple model we can find first and then expand little-by-little.
  * ***Be realistic!***

**Action Items**
 * Articles picked out (something about models or transfer learning). - **@Christian Reza @Christian Young @Spencer Thomas**
 * Put together project presentation. (**DUE 12 FEB**)
   * Outline of topics we want to present. - **@Christian Young**
     * Member introductions
     * Model types (Text-to-SQL, Q&A, Table Q&A, etc.)
     * etc.
 * Each pick out a model and investigate how it works and take notes on it. - **@Christian Reza @Christian Young @Spencer Thomas**
   * Put notes in a file in Teams. - **@Spencer Thomas**

## 16 Feb 2025

**Thoughts and feelings**
* Christian Young has been doing general research on our project and the concept of text-to-sql
  * Found a flow chart that feels expresses what we failed to express in our presentation.
    * ![text-to-sql flow chart](documentation/text-to-sql-flowchart.png)
  * Vector store seems like something we need a better understanding of - seems like its more flexible than directly querying the table.
* Spencer has asked about the EDA assignment - since we don't really have a dataset right now, following is the response from Lochana:
> Spencer Thomas: If you would like to perform an EDA for a different data set (i.e., tabular), all three of you may do that in the same data set. The reason I encourage you to analyze your LLM data set or close enough data set from the internet is that it will save you time in the end. The EDA submission and presentation itself takes 2 weeks. Even at the end, you also have to work some sort of EDA for your LLM data (once you have it) before applying to the models. 
As I mentioned to Christian Young, you don't really have to do "EDA" for the LLM data.  Maybe do a preliminary level model using an available LLM data set so that students in other teams and I know what your project direction is. I am more than happy to talk in a Zoom call if you have further clarifications.
* Would we be able to use the spider dataset as our data for our project?
  * Its a database schema [expressed as a CSV](https://github.com/jkkummerfeld/text2sql-data/blob/master/data/spider-schema.csv), it also has actual [examples of natural language queries and the transformed SQL query](https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/refs/heads/master/data/spider.json) - so it can be the actual training data we would use to transform the model.
    * *would this be too lazy?*
    * There is a [script](https://github.com/jkkummerfeld/text2sql-data/blob/master/tools/spider_schema_to_sqlite.py) in the repo that turns the CSV into a sqllite database.
* What is a T5 model?
  * Essentially, its Text-to-text-transfer-transformer
    * QA models
    * Translation models (languages)
  * Study a T5 model and find out how we can fine-tune it.
* Do we need data cleaning for our project?
  * It doesn't seem like we do, since we are not really using a text dataset, if we do it, we'd use it on the natural language queries; what we are using to train the model - *but it doesn't seem like the models need that to be trained.*
* Our assignment isn't really EDA, its more of a spike to learn about the technology and how we can achieve our goals.
* Does this make our project too easy, just transfer learning and we're done? **What are we doing that hasn't been done?**
  * What we want to prove is that we can take **ANY** dataset, apply transfer learning against a model and prove that it can do what we want it to. 
    * If we're using a dataset, we shouldn't be using a model trained off it, we want to be able to prove we can take a T5 model and fine tune it to domain-specific data.

**What are we doing for the EDA assignment?**
* ~~Use the spider dataset and do EDA on it and start looking into a T5 model?~~
* **Analyze the spider dataset schema and try to populate a database and get a text-to-sql model working**
  * Take a pre-existing dataset ([Spider 1.0](https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset/code)) and apply transfer learning to a T5 model and show what we learned and what proved to be a problem.
    * ~~OR create our own *small* schema, like 5 tables and populate manually and do a small scale test of transfer learning?~~

---- 