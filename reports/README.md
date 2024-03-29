---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [x] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [x] If applicable, play around with distributed data loading
* [x] If applicable, play around with distributed model training
* [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

99

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s190464, s214622, s214649

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used "Optuna" as our third-party framework in this project. As we've in the past semester had the course "active machine learning" we thought it would be nice to revisit Hyperparameter setting optimization with another framework. We used this GitHub as inspiration for our settings: "https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN/blob/main/optuna_optimization.py" and optimized for learning rate, optimizer dropout and experimented with a flexible depth in our FFNN. We especially found the pruning of obvious bad runs quite smart and efficient alongside the general ease of used compared to GPYOpt which we've used earlier and has very poor documentation. Optuna helped us achieve better performance and was much faster, computationally speaking, than we though it would be - An image of what information given from the Optuna sweep about HP and their importance can be found under question 14.
In future patches of this project, we would like to add more flexibility to the model architecture and make an optimization sweep over the model architecture as well.  

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

Below we've described how to set up the main files and run them, which explains the general usage of our project. If one would like to go in-depth with understanding or altering more complex or intecrit parts of the project, most of the "behind the scenes" files have comments explaining the general usage of the project.
Such, we believe that the structure layering of the project is suitable, as the complexity rises and more ML knowledge is needed to alter or understand, the deeper into the folders one has to go. - In other words: The steering of the project is relatively guided, hiding away the engine room, though it's still available to the more experienced mechanic if needed, alongside a manual.

### Prerequisition
As a prerequisite, we recommend setting up a virtual environment for this project using Anaconda.

```bash
$ conda create --name [your_environment_name] python=3.9
$ conda activate [your_environment_name] 
```
A requirements.txt file has been made in which we have stated all the libraries and packages.
```bash
$ cd [path_to_requirements.txt]
$ pip install -r requirements.txt
```

After the dependecies from requirements.txt is downloaded, the cs data can be downloaded.
```bash
$ cd MLOpsProject/data
$ python make_dataset.py
```

Lastly, the data will be cleaned, normalized and saved as dataloaders:
```bash
$ python clean_data.py
```
### Train & Predict
A new model can be trained with:
```bash
$ python train.py
```
And be used for prediction with:
```bash
$ python predict.py
```
Both train & predict files contains args that can be parsed, use --help for more info.
```bash
$ python train.py --help
$ python predict.py --help
```

### Docker
Furthermore we've made a docker file, which builds an exact image of the project and can run both the training- and predict files.
```bash
$ docker build -f train_model.dockerfile . -t trainer:latest
$ docker build -f predict_model.dockerfile . -t trainer:latest
```

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

We used the cookiecutter template given in the course. It was important for us to keep the structure of the template to force us to work in a more structured manner - We then removed the following folders at the end: The data and models in the outermost folder, as we found it more logical to have all of the things related to the model inside the "MLOpsProject" folder - Furthermore, it was also an issue with relative paths to have two folders of the same name. We also chose to remove the docker folder, as we had the docker files in the outermost folder, as we thought it would be better for reproducibility reasons. 
In the main folder, "MLOpsProject" we have everything related to models, data, training, API, and cloud. Every one of these has its folder and the main files for training etc. are in the outer-most folder, as these are the actual .py files that will be run, whereas the folder is files behind the scenes - For example imports, data, pytests etc.


### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We did try to abide to pep8 coding format, but we ended up with a somewhat altered version, yet we did abide to a set of rules between us for code quality and format, which we believe ended up being quite nice.

Throughout this project work and other projects, we've noticed the importance of using git to share code that makes for small parts in a greater whole in the project. This means that no one person has the full picture of over details in the code without reading and understanding other's code.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

The main idea was to make tests for train, model architecture and data, as these parts was changed the most throughout the project, making it important to test often. We made a total of 7 tests that tested the clean_data.py and model.py. We then used the "Coverage" library to check over coverage and reiterated our testing, both in general but also as our project grew in size.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our code coverage is 17 % (But most of the important/main files had a coverage of 80-100 %)

We think that unit-testing and thereby test-coverage in important to check for problems that might arise when growing/changing the project. But as most ML, especially DL is largely dynamic, the errors can also be very abstract. It's easy to make tests for tensor-sizes and data cleaning functions, as the answer to these tests are obvious. The real issue arises when ones loss is acting out or the accuracy suddenly is much lower after a change that even went through the tests. In this project we even had issues with data and models even at a simple FFNN - The solution was much more abstract than a simple unittest is able to handle. Therefore in ML we cannot trust test to makes us error-free, but it helps in certain areas.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We first started our workflow on the "main" branch. We quickly figured out why branching was necessary, even in small to medium projects like this. The main issue we had only using the main branch was that we all worked on this code at the same time, making the push/pull experience bad, as we had to continuously fix the merging issues.
We started adding more branches for certain situations - One for clean implementation and one for more experimental testing. For our project, this was more than fine in a 3-person group, but in bigger projects giving a branch to every person is probably the way to go.
The pull request feature was quite brilliant, as we suddenly saw zero merging issues on our main branch, meaning that the code on the main branch was always working for the actual ML testing.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We initially developed a simple version where data could be retrieved and examined using Google Drive. This proved to be a sensible decision as, during the initial process, our data was overwritten, thus allowing us to have an easy and original version that could be restored. By doing so we also made troubleshooting a bit easier. Subsequently, a Data Version Control (DVC) Pull/Push call was implemented to Google Drive with tags, enabling further examination of versions before any download.

On a more general level, errors like ours and more critical issues can be avoided by using DVC services. Data Version Control also strengthens big data flow and ensures the quality of data concerning auto-updates.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:


Unit testing: In our project, we use pytest as unit tester. The way we do this is by creating a new folder in which
three files: test_data.py, test_model.py, and test_model_structure.py are placed. These files contain functions that test different aspects of our project
like do we remove all grenades from the dataset. The first one tests all the data-related code and the test_model.py test if the model is properly running.
test_model_structure.py test the structure of our code. 

We've then added these Pytests to GitHub Actions, such that they will run on every pull request merge and any main branch push. This has helped is tremendously in finding bugs throughout evolving the project, writing more code, rewriting code, etc. We've caught quite a few bugs that would otherwise how taken time through the debugger and were in this way quite easy to solve, as we already knew the exact location by looking through the logs. 

The GitHub Action tests-run can be found here: "https://github.com/GorenzelgProjects/MLOpsProject/actions/workflows/ci.yaml".

We did however not evolve our pytests as much as we would have liked throughout the project. We added tests model, data, and training as we though these were bare-minimums but as the project grew, we would have liked to have had the time to implement even more. - In further patches of this project or other projects, we will prioritize implementing more CI, as we have been able to see the benefits hereof. - Even though it can be tedious to set it all up, once it's running, it can save one a lot of time.


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We mainly used Hydra to configure our experiments, which has all its parameter settings in a config file and saves all important experiment data in an output folder with timestamps.
As an addition, we've also made a "click"-library-based file for command line training and prediction, where one can input settings directly into the command line. Question 4 showcases some of these command-line coding examples for running experiments.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

For experimental reproducibility, we used both Hydra and Wandb to store important information from our experiments. Hydra took care of the config files and saved a local file every time we experimented with hyperparameters, data information, loss, and the trained model. This was enough to let us reproduce the experiments if needed. However, we lacked the online functionality of Wandb, so everyone on the team could follow training in real-time as it was being made. Wandb also added the ability to share loss, accuracy etc.-plots on the go, making the inference analysis much easier as a team. The mixture of these two tools went a long way in making reproducibility easier and safer and we were in general very happy about working with increased reproducibility. 

We added different workflows in the code, such that we in a code/ML-debugging scenario could stop and start the tracking if need-be.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

![my_image](figures/wandb_main.png)
![my_image](figures/wandb_val_loss.png)
![my_image](figures/Optuna.png)

As seen in the first image, we've tracked the train+validation loss and train+validation accuracy. These two factors explain both sides of a common issue in ML: exploitation vs. exploration or just generalization. The train acc. and loss explains how well the model adapts to the training set and validation acc. and loss explains how well the model generalizes to new data. -> At least that's the idea. As seen in the train loss, we have a continuously falling, relatively smooth curve throughout all the steps. As for the validation loss, we see a more bumpy, yes continuously falling curve. This suggests a few things: firstly the model is learning well in terms of the train set, yet still able to generalize well. We only see a slight overfitting even over 40 epochs. Secondly, we want to use the validation loss to force an early stop if the validation loss stagnates or even rises over a few epochs. This didn't seem necessary over 40 epochs as it was continuously falling, but if we wanted to go for example 100 epochs this should be implemented.

This explains that even with just 4 loggings of val/train loss and accuracy, we get a lot of information on what to do next - And this is just the tip of the iceberg. We were very happy to implement W & B, as it was easy to go back to each experiment and either use the information to improve or go back to a previous (better) version, as it also saves HP settings and model architecture.

The last image is a HP-sweep of our network using the framework. We opted to go with this over W&B, as we wanted to try this new framework. More information about the Optuna sweep in Question 3.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

Docker has also been a focus of interest in our project, both on its own, but also to use alongside our API and Google Cloud implementation. Separately we used two docker images: one for training and one for inference. We mostly ran the images out of the box, but it is possible to change hyperparameter settings by changing the Hydra config file. Link to training file "https://github.com/GorenzelgProjects/MLOpsProject/blob/main/train_model.dockerfile".
Furthermore, we used docker images for our API deployment, as this would make sure we could run our solution on any PC, just by running the docker file.
Lastly, we also used a docker image for our Google Cloud setup.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

In our project, we used Visual Studio Codes debugger, which helped a lot in getting our project to work. We found this technique to be very time-efficient for debugging plus efficient code checking (run sub-parts of the code). Using the breakpoint function and debugger console allows us to move beyond print-statements and we found this very efficient. However, the debugger tool still works best for "old-school" code problems, whereas other tools like plots and shape prints are still necessary for ML issues/bugs debugging.
Already in the early project days, we focused on implementing profiling to see which parts of the code were slow. This helped us speed up the code, as we now had a better idea if where we could improve the most.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We did not get a fully built/running cloud model, but we did make use of the following:

Cloud Storage: This service was used for the storage of data, models, and trained models. This is done by having our buckets here. Furthermore, this is a necessary service to use if we would use Google Cloud.

Container Registry: Here we saved our docker images, created from our local computers.

Compute Engine: This provided us with WMs that we should use if we wanted to run code on Google Cloud.

Vertex AI: This is a place where it is possible to quickly get an AI model and pre-build a container going to some data. Later in the project, we could have been using the custom option to get our code working on the cloud with custom containers. (We tried a lot...)




### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

As mentioned in Question 18, we did not get to have success with having a working implementation, but we did however build out docker images on our computers, which were pushed to the cloud. On the cloud, we tried to build up an instance to run this with VMs. For experiment purposes, we used the minimal hardware available to us. Of course, we had made a custom container for this. That should be said, we did spend a long time on this and we should probably have allocated even more time to this, as we found this topic harder than anticipated (even after Nicki's warning in class).

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![my_image](figures/bucket1.png)
![my_image](figures/bucket2.png)
Our trained model is placed in the Google cloud folder.

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

We tried a lot with different settings and versions. We should probably have sorted this part a bit more. Inside the different folders are the images we created.
![my_image](figures/Containerregistry.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![my_image](figures/cloud_build_history.png)
We build our images locally on our computers, therefore there is nothing in this history. But we can write a bit about what Cloud build is:

On Google Cloud you can upload source code from a computer. From the code, Google Cloud then builds a container to this code, where you can specify a lot of settings. The result is an artifact, like a docker container in our case. Furthermore, this function also can provide an extra security for a system chain. Cloud Build would ideally have been one of the important aspects of a working application chain.

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:


We succeeded in deploying our model locally. To achieve this, we built a Docker image around our main file, utilizing Fast API as our web framework. The main concept is that users can interact with a trained model, providing inputs about information for a Counter-Strike round, and the model will then return a prediction based on the user's input on who will win the round. However, we did try to implement it on the cloud as mentioned before.

Ideally, we would have liked to have a functioning application running on Google Cloud because we then could have scaled our training a lot. Furthermore, if we had access to a larger database of those .csv files, we could have used the storage function to easily store it and then used it to train. This model could then have been trained on a much faster GPU, which means something when training on huge datasets.  

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

As mentioned before we didn't succeed in deploying our model, but monitoring of deployed model is important, here is why:

Firstly, it helps to monitor the actual performance of a deployed model over time. This can be given in a lot of different ways as matrices and so on. Secondly, it makes it easier for us to control the data, which are going through the model. It could be cases of data drifting or something unusual/outliers, which should be dealt with. In our case data drifting would certainly happen, as with game updates, the data would change over a period and we would expect to gather new data and retrain at at least every patch. Thirdly, there is the hardware part of training a model, like how is the memory usage and tracks CPU/GPU. 

Due to these opportunities, it would have been nice to have and we could see if our model was deployed to the cloud that this should be a fundamental part of us using the cloud.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

In total, we used 240 kr. which was given in the free trial version. This was mainly tried one computer, therefore it is only one of us there used that. If we had implemented our model in the cloud and scaled up, as we wanted, it would have cost a lot more. 

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![my_image](figures/overall.png)

The starting point of the diagram is our local setup, where we integrated a Pytorch Feedforward Neural Network (FFNN) application for predicting Counter-Strike round winners. This application is controlled, optimized, and logged respectivly using HYDRA, Optuna, and Wandb. Additionally, we have a Fast API user interface named "main," and the local deployment is encapsulated within a Docker Image, providing the capability for training, prediction, and API calls.

To manage dependencies, we utilize Anaconda Prompt for efficient pip install operations. Simultaneously, we integrate Git commands, such as git push and git pull, to synchronize our local codebase with the GitHub repository. Furthermore, we incorporate Data Version Control (DVC) to handle data versioning, utilizing dvc pull and dvc push to interact with a Google Drive storage for seamless collaboration and version control.

Whenever code is committed and pushed to GitHub, it triggers GitHub Actions, a part of our GitHub setup. This includes profiling and debugging tools such as Pytest and cProfile. The GitHub Actions automate various processes, ensuring code quality and reliability in our version control system.

The overarching idea of allowing users to automatically upload log files, make an API request, and receive real-time recommendations could be realized through the use of Google Cloud products if further development time was available. As a starting point, we would have used the Google Clouds Run function to run some of the Google Images we created early on in the process. Where the Images were stored in the Container registry. This could then give us ensure about whether our model would work in this way of handling it or not. We would also have liked to use our model on Vertex AI for cloud based ML. This would have made it possible for us to run it on some hardware, that we do not necessarily have available. Furthermore, we would have liked to utilize a solution involving Google Cloud Bucket and integrated the data into the cloud environment.

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Throughout the process, we had different challenges. Firstly, we had program issues with our model to work on our data. This needed a mixture of debugging, profiling, and wandb to solve it. Debugging helped with issues regarding folder layout giving data-loading issues, profiling to speed up very slow parts of the code, and wandb+shape-print-statements to fix model issues. Hydra also gave us issues with config files in the training loop, but in the end, the tools we've gotten both before this course and throughout it helped us overcome most of the local deployment issues and made for a good ground for reproducible experiments 

Secondly, we had the Google Cloud issue (surprise, surprise) which took a huge amount of time working. To solve the first problem we had to look through several videos on YouTube and experiment with code before it worked. For the second problem, we searched through the internet/YouTube to solve it, and it seemed to be a huge struggle in the beginning. The first couple of steps was fine in which we mean creating a project and activating API and bucket. But from there it was time-consuming to debug through the cloud - As explained in the Cloud questions, we should have allocated even more time for this part.

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Student s214649 was in charge of developing the cookiecutter structure and the initial github setup. Furthermore implementation of github actions. He also did Hydra and Wandb setup in the pytorch networks.

Students s214649 and s190464 was in charge of developing the pytorch models, train- and test-loop.

s190464 was in charge of developing the docker files/images, DVC- alongside the fastAPI implementation. He also did the graphical designs of the project.

s214622 was in charge of developing the pytests-files, profiling, coverage, and everything about Google cloud (Buckets, Engine, Builds, Clouds runs and Vertex AI), including its overall idea in the MLops stack.

All members contributed to the report and are familiar with the code, even though they did not necessarily write it themselves.
