# Use GANs to generate art work 

currently uploading files, I'll update the readme.md soon


The goal of this project was to use GANs to reproduce art work as faithfully as possible.

## Dataset
The dataset provided by the artist was composed of 222 very high quality black and white images. 

## Early attemps 
In a first step, I tried to train a GAN on this dataset, but the results obtained were not conclusive, even after optimizing the hyper parameters. 
The next attempt was to use data augmentation techniques, but it did not produce images that could have been made by the artist. 

# Final approach
The final approach consisted of two major steps: 
- using various autoencoders to increase the size of the starting dataset (convolutional and feedforward autoencoders)
- to train a gan on this dataset of size 666 



![Original Images](https://github.com/gjmichel/gans-in-art/blob/main/results/original_images.jpg)

![Reconstructed Images](https://github.com/gjmichel/gans-in-art/blob/main/results/reconstructed_images_feed_forward.jpg)

![Generated Images](https://github.com/gjmichel/gans-in-art/blob/main/results/images_being_generated.jpg)

![Generated Images](https://github.com/gjmichel/gans-in-art/blob/main/results/generated_images.jpg)
