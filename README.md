## A Quick Note
This code was developed by me as a kind of side project which served two purposes (besides of course obtaining some form of result):
1) Gain some experience with writing Neural Networks in Matlab
2) Considering I am already confident in my ability to produce code for Neural Networks in Matlab that works properly, but at the time was not as comfortable with developing in a Python environment, I wanted to use this as the "base code" and re-write the whole thing in Python. You can find the Python code in my PYTHON-Cognitive-State-Detection repository.

Considering that this particular code is not intended for serious use, and considering that I was developing on some not-so-stellar hardware at the time and neede to train the algorithm 40x9 times (40 individuals to learn on, tabulating results from 9 training sessions to ensure consistency), I limited the number of iterations during the training sequence to 100 **(the Python Neural Network was also subject to these constraints)**. This was enough time to train the network on some datasets, whereas in others it was not. Some things to keep in mind:
* Some data grouped into linearly seperable classes. In these cases, 100% (or close) accuracy should be expected.
* Some data was much more messy, the test accuracies for these cases are expected to be relatively low because either the 100 iteration training period was not enough time for the optimization function to converge, or simply because the data was too messy to learn on
* Some data was very poor and for these test accuracy should be expected to be low, a longer training session may not fix this.

**To see how the Neural Network performs after a sufficiently timed training session, see the test accuracies listed in the TENSORFLOW-Cognitive-State-Detection repository.**

# Introduction
My graduate advisor worked in the past with a prominent U.S. entity (which shall remain nameless) which was interested in monitoring the cognitive context of its employees through non-invasive means in order to better delegate tasks and reduce human error during human/machine interaction. Previous studies have related pupil diameter to the level of focus which an individual is currently experiencing. The entity had developed a test "game" which contained three categories of tasks: easy, somewhat difficult, or very difficult. The goal was to collect pupil diameter data recorded throughout the test from various individuals and accurately predict when the individual was very focused vs. not very focused - assuming that the majority of people would be very focused during each very difficult task and vice-versa for the easy tasks. In practice however, the mean and standard deviation of each participant's pupil diameters during each task hardly show any differences during each task. Results were somewhat lackluster and little was done to push the project forward. I thought that I would be able to improve on their results by measuring alternative characteristics of the data and performing a few feature enginering steps before finally feeding input data through a Feed Forward Neural Network.<br/>

This repository contains a program which employs a Neural Network (NN) designed to recieve input data in the form of an individual's pupil diameter in order to detect whether or not the particular individual is currently engaged in a task which they consider to be easy, somewhat difficult, or very difficult. A short training/calibration period is necessary (approximately 20 minutes - the duration can be adjusted depending on necessity), during which time the NN learns the specific parameters of that individual. After the training period the program is designed to take measurements every 2 minutes using data from that time frame to construct a single feature vector, outputting the level of focus that the individual was most likely experiencing during that two minutes. 

# Applications
This was a personal project, and work with the entity that was originally interested in the technology has long been discontinued - which is unfortunate because the functionality of the Neural Network approach compared to their previous attempted solutions offers several benefits. If I were to pitch a situation in which this technology might be useful I'd outline the following scenario:<br/>
* Consider that you have a small workforce to which you must delegate a handful of tasks in teams. 
* Using a similar test "game" which contains tasks that can be reasonably considered easy, somewhat difficult, and very difficult for the majority of people, have each individual seated at their workstations complete the game to calibrate the program during the ~20 minute training period. You now have each individual's unique metrics at each of the three difficulty levels.
* Once the program is calibrated, have each idividual perform each of the potential tasks while the program runs in the background, monitoring their levels of focus.
* Compare results. Say Angela, Dwight, and Jim perform task X with little effort as evidenced by the program predicting their metrics during this task as most similar to those collected during the "easy" task at test time, while the rest of the workforce was predicted to have struggled somewhat. Perhaps assigning this job to Angela, Dwight, and Jim would result in higher productivity and fewer mistakes. Each task can be assigned their ideal teams.

# Function/File Descriptions
* **MatlabResults:** A csv containing the results from 10 trials on each participant. 40 participants total.
* **aMain:** The main program file. All helper functions are called here. Some data parsing/feature engineering takes place in this program file but the majority is handled through smaller functions.
* **createTrainSet:** A function used in order to parse the data into three distinct training sets so that labels may be added. Sets are later combined back into on single training set with the appropriate labels. Also works for test sets.
* **data:** Contains data from all participants.
* **fmincg:** An optimization function using the conjugate gradient method. Used to minimize theta values given by a cost function's gradients.
* **nnCostFunction:** The cost function for the neural network. Performs both forward propagation steps to compute the cost as well as the backwards propagation step in order to compute the gradients.
* **predict:** Used to run predictions on the test set once the optimum parameters for the neural network have been determined.
* **randInitializeWeights:** This is required to randomly initialize the Theta parameters prior to training, to avoid symmetry.
* **remove_outliers:** A short function which detects outliers from the data and removes them.
* **sigmoid:** The activation function in each node of the hidden layers and output layer of the network. 
* **sigmoidGradient:** Used in "nnCostFunction" in order to compute the gradients of the weights for optimization.

