# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?
A: *Naked Twins is a Sudoku strategy which places further constraints on possible solution which allows to eliminate potential playing possibilities. This makes game search space smaller and thus allows to find the game solution faster.
Naked twins strategy looks for two boxes in the same unit which both have the same two possible digits. We can conclude that these digits must be in these boxes although we dont know which one where, but we can eliminate these digits from every other box in the same unit.
I implemented Naked Twins by searching for every box with 2 digits in particular unit and finding the box with the same pair of digits. I stored the twin boxes in a list in a hash keyed by the 2 digits they share. Finally, I've removed both digits from every one of their common peers.*

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?
A: *Diagonal sudoku does not require the modification of the actual constraint propagation algorithm. It stays exactly the sam. The reason is simple: Diagonal sudoku merely expands number of game units that need to be taken into account, hence the core algorithm did not need to be modified. The only thing that needed modification was the code that expands the unit space with the diagonal units of boxes. This was done simply by crating particular box coordinates and concatenating them to the original units.*

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

## Docker

There is a Dockerfile attached to this project which allows you to build a docker image and thus keep your workstation clean and your work evnrionment consistent and portable.

Build docker image:

```
docker build -t aind .
```

Once you have your image built you need to mount your working directory into the container:

```
$ docker run --rm -it -v FULL-PATH-TO-SRC-DIR:/udacity aind /bin/bash
root@1d6677f07c8e:/# cd /udacity && source activate aind
(aind) root@1d6677f07c8e:/udacity#
```

Any changes you make on your workstation are immediately available in your Docker container.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the `assign_value` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login) for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

