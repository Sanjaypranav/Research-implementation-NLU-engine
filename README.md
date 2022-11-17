<!-- version tag -->

<div>
<a href="https://colab.research.google.com/drive/1ghKnhJG1uXlA93y6trjy4hDsVzMLidhw?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
 <a href="https://github.com/prakashr7d/Research-implementation-NLU-engine/tree/main/.github/workflows"><img src="https://img.shields.io/badge/ruth-passing-green?style=flat" alt="ruth" /> </a>
 <a href = 'https://pypi.org/project/ruth-python/'><img src="https://img.shields.io/badge/ruth-PyPi-blue?style=flat&logo=python" alt="ruth" /> </a>
 <a href="https://medium.com/@Sanjaypranav/ruth-a-natural-language-understanding-framework-developed-by-puretalk-ai-5f229aacbf2a"><img src="https://img.shields.io/badge/read-medium-lightgrey?style=flat&logo=medium"></a>
 </div>

<img src="https://www.puretalk.ai/static/assets/Puretalk.png" height= 200px>
 
Ruth Natural Language Understanding
---
Welcome to RUTH (Really Understanding The Humans) NLU documentation. RUTH is a open sourced Natural Language Understanding (NLU) framework developed by [puretalk.ai](https://puretalk.ai/). It is a Python module that allows you to parse natural language sentences and extract information from them.

RUTH is cli based tool that can be used to train and test models. 


# <div align="center">Installation </div>


# Quick installation
    
    $ pip install ruth-python

### Installation from source

    
    $ git clone https://github.com/prakashr7d/Research-implementation-NLU-engine.git
    $ cd Research-implementation-NLU-engine
    $ python setup.py install
  
### Using makefile (for linux & mac users)

 [Makefile](https://www.gnu.org/software/make/manual/make.html#toc-Overview-of-make) is a file that contains a set of directives used by make build automation tool to generate executables and other non-source files of a program from the program's source files.


    $ git clone https://github.com/prakashr7d/Research-implementation-NLU-engine.git
    $ cd Research-implementation-NLU-engine

for ubuntu, 

    $ make bootstrap
   <!--create's poetry  -->

for mac,

    $ make bootstrap-mac

then finally to install package run the following bash command

    $ make install


### Pytorch installation with GPU support


    $ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116



# <div align="center">Documentation </div>


 Getting Started
----
The main objective of this lib performs to extract information by parsing the sentence written in natural language. To getting started with RUTH let's follow the below steps.
Run the following command to build an initial project with data and a default pipeline file.

    $ mkdir project_name
    $ ruth init 

Output

Project will be initialized with below structure 
```bash
.
├── data
│    └── example.yml
└── pipeline.yml
```
Project will be created with example data and pipeline   

## <div >CLI </div>

RUTH has a CLI interface to train and test the model, to get started with the CLI interface, run the following command

    $ ruth --help
    
for example, to train the model, run the following command

    usage: $ ruth [-h] [-v] {train,test} ...

## <div >Training </div>


To train the model, run the following command

    $ ruth train -p path/to/pipeline.yaml 
      -d path/to/dataset.json

Parameters


    -p, --pipeline  Pipeline file 
    -d, --data dataset path 
 Saving Trained models
---

Once the training is finished the model will be saved in a directory named `models` in the current working directory.

Dataset format 
---
RUTH uses a yaml file to store the training data, the yaml file should have the following syntax

example 

```yml
version: "0.1"
nlu:
- intent: ham
  examples: |
    - WHO ARE YOU SEEING?
    - Great! I hope you like your man well endowed. I am  &lt;#&gt;  inches
    - Didn't you get hep b immunisation in nigeria.
    - Fair enough, anything going on?
    - Yeah hopefully, if tyler can't do it I could maybe ask around a bit
- intent: spam
  examples: |
    - Did you hear about the new Divorce Barbie? It comes with all of Ken's stuff!
    - I plane to give on this month end.
    - Wah lucky man Then can save money Hee
    - Finished class where are you.
    - K..k:)where are you?how did you performed?
```

<div >Pipeline</div>
---
RUTH is a pipeline based NLU engine, it has 3 basic main components
- Tokenizer
- Featurizer
- Intent Classifier

In [pipeline-data.yml](https://github.com/prakashr7d/Research-implementation-NLU-engine/blob/main/data/test/pipelines/pipeline-basic.yml) file is used to define the pipeline and its components
Example of pipeline-basic.yml file for Support Vector Machine (SVM) based intent classifier and CountVectorizer based featurizer.

```yaml
task:
pipeline:
  - name: 'WhiteSpaceTokenizer'
  - name: 'CountVectorFeaturizer'
  - name: 'SVMClassifier'

```

```yaml
task:
pipeline:
  - name: 'HFTokenizer'
    model_name: 'bert-base-uncased'
  - name: 'HFClassifier'
    model_name: 'bert-base-uncased'

``` 
## <div >Parsing </div>

To parse the text, run the following command

    $ ruth parse -m path/to/model_dir 
      -t "I want to book a flight from delhi to mumbai"

Parameters

    -m, --model_path  model file (optional)
    -t, --text  text message (required)


If model path is not provided, Parse function will use the latest model in the model directory as a default model.

## <div >Testing </div>

To test the model performance, run the following command

    $ ruth evaluate -p path/to/pipeline-basic.yml 
      -d path/to/dataset

Parameters 

    -p, --pipeline  pipeline file 
    -d, --data  dataset file
    -o, --output_folder  to save result as PNG file (optional)
    -m, --model_path (optionol)

If model path is not provided, `Evaluate function` will use the latest model in the model directory as a default model. If output folder is not provided, the result will be saved in `results` folder in the current working directory.

### <div >Deployment </div>
RUTH uses FastAPI to deploy the model as a REST API, to deploy the model, run the following command

    $ ruth deploy -m path/to/model_dir
Parameters

    -m, --model_path  model file (required)
    -p, --port port number (optional)
    -h, --host host name (optional)

Output

```bash
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:5500 (Press CTRL+C to quit)
```

## <div >API </div>

Once the model is deployed, you can use the following API to parse the text

    POST /parse
    {
        "text": "I want to book a flight from delhi to mumbai"
    }

Output

    {
        "text": "hello ruth!",
        "intent_ranking": [
            {
                "name": "greet",
                "accuracy": 0.9843385815620422
            },
            {
                "name": "how_are_you",
                "accuracy": 0.0017248070798814297
            },
            {
                "name": "voice_mail",
                "accuracy": 0.0008955258526839316
            },
        ],
        "intent": {
            "name": "greet",
            "accuracy": 0.9843385815620422
        }
    }

<!-- for social connects -->
##### <p>Connect us with </p>


[<img align="left" alt="Puretalk | LinkedIn" width="30px" src="https://img.icons8.com/color/48/000000/linkedin.png" />][linkedin][<img align="left" alt="Puretalk | Twitter" width="30px" src="https://img.icons8.com/fluent/48/000000/twitter.png" />][twitter][<img align ="left" alt="Sanjaypranav" width="30px" src="https://img.icons8.com/color/48/000000/gmail.png" />][gmail]


[linkedin]: https://www.linkedin.com/company/puretalkai/
[twitter]: https://twitter.com/puretalka
[gmail]: mailto:info@puretalk.ai
<br>
<br>
Devoloped by Puretalk@2022 
<br>
<!-- <img src="data/img/ruth_g.png" width=50px height='50px'> -->
