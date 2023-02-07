# CAIS-Winter-Project
Jessica Fu
fujessic@usc.edu

In this project, I applied ResNet-50, a pre-trained computer vision model, on a preprocessed
ASL Alphabet dataset from Kaggle to perform multiclass classification of the American Sign
Language alphabet. The ASL Alphabet dataset consists of 87,000 images that are 200 x 200 pixels.
There are 29 classes: 26 for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING. The
test dataset has 29 images, one for each class.

For preprocessing, I scaled the images to 224x224 pixels and normalized them since I was
using a pretrained model and the input needed to match the format on which the network was
originally trained. Normalizing the images also helps convergence occur quicker during the training
of the neural network, producing faster results.

When training the model, I referenced another Kaggle notebook, “ASL Alphabet
classification PyTorch,” created by Julia Ponomareva in addition to the Transfer Learning notebook
from the CAIS++ curriculum. I decided to use the pre-trained ResNet-50 model with transfer learning
because the model is most commonly used for image classification and recognition tasks and has been
extensively trained on a diverse range of image types from the ImageNet database. Transfer learning
allowed for much lower computational and time requirements compared to if I had trained a CNN
from scratch and allowed me to work with a smaller data sample size.

Due to computational and time constraints, I only used 30,000 of the 87,000 training samples.
I put 80%, or 24,000 random samples, into the training set and 10%, or 3,000 random samples, into
the validation set and the remaining 10%, or 3,000 samples into the test set. I decided to increase the
number of test samples because it would allow for a more accurate evaluation of my model’s
performance since the original test set didn’t cover variations within each class such as the ASL sign
in bright light or dark shadows. Since I was using transfer learning on a smaller dataset, I decided to
freeze most of ResNet-50’s pre-trained layers to prevent overfitting and fine tuned the fully connected
layers to output 29 classes. This also made the training more efficient as not all layers would need to
be updated based on the ASL dataset.I defined theloss as Cross Entropy Loss, which presents the
correctness of the model’s predictions as probability, a good fit for classification tasks, and chose
Adam as the optimizer, since it is fairly simple to tune, performs well even with noisy images, and is
computationally fast. I set the learning rate for the optimizer to 0.001 which results in more optimal
weights compared to a higher learning rate despite making the training process longer. I made my
batch size 32 because it’s pretty standard, being neither too small nor too large.I tested out the
training with 2, 4, 6 and 5 epochs, with 5 being the most optimal.

For the model metric, I used accuracy which is the percentage of correct predictions out of
the total predictions. The best validation accuracy was 0.9877, and the lowest validation loss was
0.0579. Though the accuracy was high and the loss was low for the validation, and the predictions
were very accurate with the test set, when the test set images’ colors were oversaturated, the model
could not correctly identify most of the ASL signs. This was in part due to the range of images
available in the training set, with all of them having very similar backgrounds and the only major
variation being the lighting on the hand.If the datasethad included images with more distorted
perspectives, with different backgrounds and varying amounts of noise, and with different skin-toned
hands and different lighting, it would have allowed for better generalization.Furthermore, if I had
more computational resources and power, I could have increased the number of data samples for
training, which could help decrease overfitting.Icould have also tried to decrease the learning rate
and subsequently increase the number of epochs to improve the model accuracy.

Though this project only focused on classifying the ASL alphabet, if the classification
techniques were extended to ASL as a whole and combined with NLP to enable the translation of
ASL to written English and vice versa, it would greatly benefit those who use ASL to communicate
and promote inclusion.Such technology could benefitthese individuals in their everyday lives like
communicating with co-workers, ordering food at a restaurant, and more. Furthermore, currently
some airlines like Southwest don’t have video recordings of safety procedures with captions, so flight
attendants would verbally communicate these procedures which can cause difficulty for individuals
with hearing loss. It can be very beneficial if such individuals had access to translation software in
situations like these.
