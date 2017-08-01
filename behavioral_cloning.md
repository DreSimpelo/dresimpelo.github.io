---
layout: page
mathjax: true
permalink: /behavioral_cloning/
title: Behavioral Cloning
---

<div class="youtube">
<iframe width="560" height="315" src="https://www.youtube.com/embed/-8h2U6tdlqA?rel=0" frameborder="0" allowfullscreen></iframe>
</div>

For this project, we had to predict steering angles given (virtual) dashboard camera images. One way to approach this, would be to get a computer to mimic a professional human driver (me!). This is done by logging corresponding steering angles with their temporally paired camera image. A convolutional neural network is a perfect candidate for dealing with pixel data, so lets begin! 

## 1. Model Architecture

I decided to base my architecture off of the Nvidia End to End Learning for Self-Driving Cars [paper](https://arxiv.org/pdf/1604.07316v1.pdf). My architecture also has some design inspiration from [VGG](https://arxiv.org/pdf/1409.1556.pdf) as well as utilization of [Batch Norm](http://jmlr.org/proceedings/papers/v37/ioffe15.pdf) for ease of training and simplicity (Details below). 

I'm a fan of the elegant/clean nature of the VGG architecture. I understand it is no longer state of the art, but I still like their philosophy of simplicity.

I'm also a huge fan of Batch Normalization. It makes weight initialization less important, and also improves the quality of the gradient flow during backprop (due to forcing our activations to have a normal distribution) ([Slide 69 cs231n Lecture 5](http://cs231n.stanford.edu/slides/winter1516_lecture5.pdf)). I added this in hopes of faster training, and a cleaner gradient flow.  

**Overview of Architecture:**

- I use rectified linear units as my default choice of activation function throughout my network. It is a battle-proven, easily interpretable and a go to choice for many state of the art networks. 

- I follow each convolutional and fully connected layer with a Batch Norm layer. 

Layer 1: Normalization.
   - Normalize the pixel data to have a range between 0 and 1.

Layer 2-5: Convolutional Layers

  - I decided to follow the Nvidia paper with filter sizes, strides, and number of layers. 

  - **Nvidia starts with 3 5x5 filters, with stride 2x2.**
  - Conv1:
     - I start off with 64 5x5 filters, with stride 2x2. I decided to start with 64 filters since it is a clean power of 2. I probably could have gotten away with using 32 filters (as Nvidia uses only 24 filters in their real-life model).
  - Conv2: 
     - Similar to Conv1, I use 64 5x5 filters.

  - Conv3: 
     - I double the amount of filters to 128 at this layer (*inspired by VGG*), still using 5x5 filters and 2x2 stride (Nvidia)

  - **Nvidia follows by switching to 3x3 filters, with normal strided convolutions for the last 2 convolutional layers**
  - Conv4: 
     - 128 3x3 Filters. (Simply keeping the filter count the same as the previous layer)
  - Conv5:
     - 256 3x3 Filters. (Doubling the amount of filters of the previous layer)
  - **Nvidia follows the convolutional layers with 2 fully connected layers**
  - FC 1:
     - The previous layer is flattened into a long vector, and used as input to the fully connected layer.
     - We have 1024 units in this 1st FC. 1024 is a random number chosen (Ball-park guess of what is enough) to ensure we have enough representaional power for this regression model.
     - We use 0.8 Dropout here to prevent overfitting. 

  - FC 2:
     - We finish with 256 units, before predicting our steering angle. 

- **Output Predicted Steering Angle**  

## 2. Creation of the training dataset and the training process

During the process of data acquisition, I was unsure of what kind of data would be needed to successfully drive both tracks. So I recorded a single dataset for each of the tracks, trained it through a few models [all models are based off of the 'base' model listed above], and evaluated them through a critical review of its performance through the simulator. 

I found that it is very easy to train a model to memorize a single track (either 1 or 2), but it is a bit more challenging to make a model to do well on both tracks. I also did not just want to memorize both tracks, I wanted the car to be as truly autonomous as possible.

I found that simply driving through both tracks wasn't going to cut it (at least with my models). 

For example, the car would successfully drive through some parts of track 1, but fail horribly at other parts. At other times, the car would simply drive off the track and into the ocean, or take the dirt path after the textured bridge, which is usually unsuccessful. At other times, my model would be able to fully drive laps around track 1, and fail horribly at track 2.

Through my experiments, I found these obsticles to be the toughest to tackle:

<div class="fig figcenter fighighlight">
<img src="/assets/behavior_cloning/fig8_hardest_turn.png"> <div class="figcaption">
Figure 1: This is the turn where my car would drive off the track, and into the ocean most frequently.
</div>
</div>

<div class="fig figcenter fighighlight">
<img src="/assets/behavior_cloning/fig7_challenge_turn.png"> 
<div class="figcaption">
Figure 2: This is where my car would drive into the sand part, and fail afterwards. I could have introduced "sand-driving" recordings to combat this edge case, but my current model is not supposed to drive on dirt.
</div>
</div>

## Avoiding overfitting

I took some measures to avoid points of overfitting as much as possible. 

**Theory:** I think that when using the full (160,320,3) image to drive our car, it makes it easy for our car to pick up on queues to memorize. It doesn't take many epochs to get a model that can 'autonomously' drive around a single track (Trained and Evaluated on a single track). And it is still hard to quantify/interpret how much memorization was going on even when introducing heavy dropout to reduce the memorizing/overfitting.

**Proof:** (At least for my model) I recorded a seperate dataset that drove through the track in the ***backwards*** direction (Inspired by Mario-Kart), and tried to see how well my model would do driving forward (regularly). My model was able to make some simple turns, but would fail very quickly driving off track. This shows us that for my particular model, it was indeed memorizing some artifacts in the environment to help it drive successfully in the previous experiment (trained and evaluated with forward driving). 

To combat this proposed problem, I cropped off the top half of all images, leaving me with images that dont take into account environmental queues to memorize (See figure 3). 

<div class="fig figcenter fighighlight">
<img src="/assets/behavior_cloning/fig3_full.png"> 
<div class="figcaption">
Figure 3: Before Crop [Shape: 160,320,3]. We have access to all environmental queues, which I argue are easily memorized 
</div>
</div>

<div class="fig figcenter fighighlight">
<img src="/assets/behavior_cloning/fig4_cropped.png"> 
<div class="figcaption">
Figure 4: After Crop [Shape: 80,320,3]. We can see the model gets enough information of the road to make a steering decision, without having access to environmental memorization queues
</div>
</div>

**Problems:** We could argue that cropping the image is not realistic, in the sense that in the real world, we will want our models to be able to use environmental queues to make decisions. In this simulation, I make the assumption that the objective is just to safely drive around the track. And to get an honest exerpience of autonomously driving (as least memorization as possible), I decided to crop the top half of the input camera image in order to prevent memorization of environmental queues around the track.

Using the data from the recording session of driving through the track ***backwards***, I trained a model with the cropped version of the images, and evaluated the model on driving forward through the track. My model was able to drive laps autonomously through the forward path. I argue that very little memorization is leveraged to successfully autonomously drive.

I make an (unproven) assumption that it is now "safe" to use data from driving forward, as long as its cropped. 

To further avoid overfitting, I utilize dropout after the large fully connected layer. 

## Recording Session and hyper-parameter details

Using data from 5 different recording sessions (Detailed below), I trained my model with the following parameters: 
   - Adam Optimizer: lr=0.0001 -> (15 epochs) -> lr=0.00001 -> (5 epochs)
      - I drop the learning rate to 0.00001 because in experiments, 15 epochs was enough to converge. And the extra epochs from the dropped learning rate were successful in dropping validaion cost. 

5 Recording Session Details:
   - Recording Session 1:
      - Driving through track #1 4 times, straight driving, slalom driving, hugged left line, hugged right line.
   - Recording Session 2:
      - Driving through track #1 2 times in the backwards direction; straight and slalom driving. 
   - Recording Session 3: 
      - Driving down track #2 to the end, center driving. 
   - Recording Session 4: 
      - Driving down track #2 backwards, from the end to the start.
   - Recording Session 5:
      - Driving down track #2 forwards, hugging outter lane during sharp turns (to avoid hitting/turning into walls during sharp turns)

# Training process

I leverage a small validation set (less than 5% of the data), just to monitor any possible overfitting of the dataset. I kept an eye out for a stable validation loss, when the validation loss would spike, I would assume overfitting is occuring. I trained using shuffled batches of 128. That is, we are assuming our model is making frame by frame predictions, since we train with random batches of 128 images. I started off with 0.0001 Learning rate for 15 epochs, which was enough for the cost to converge, and followed that with 5 extra epochs of a dropped 0.00001 learning rate. This further dropped the cost, as well the validation cost. After this, we exported the keras architecture and model weights and dove into the simulator.

# Final notes
My model is able to successfully drive through both tracks. 

(NOTE: The 2nd track starts off weird, where the car struggles to move up the hill. If you wait a couple of seconds, the car will have enough force to make it up the hill and [hopefully] autonomously drive).
