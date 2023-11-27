# classifier-latent-diffusion

Basic idea here:
-Take an off-the-shelf diffusion model, 
-"unsample" some images using reverse DDIM,  
-train a classifier on those unsampled images, 
-look for adversarial examples in that space

The hunch is that the diffusion model will distort the space in such a way that it's easier to generate images that look like the actual adversarial target class than to add in weird high frequency noise that fools the classifier while leaving the image unchanged.
