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
 * Each pick out a modle and investigate how it works and take notes on it. - **@Christian Reza @Christian Young @Spencer Thomas**
   * Put notes in a file in Teams. - **@Spencer Thomas**


---- 