# Basic_terms
To define and create Neural Network

1. Model Deployment: 
- Inference: The process of using a trained model to make predictions or generate outputs on new, unseen data. 
- Model Serialization: Saving the trained model's parameters and architecture to a file format for future deployment and use. 
- Model Compression: Techniques to reduce the size of the trained model for efficient storage and deployment on resource-constrained devices. 

2. Transfer Learning Techniques: 
- Fine-tuning: Updating the parameters of a pre-trained model on a new task or dataset by further training it with the new data. 
- Feature Extraction: Using the pre-trained model as a fixed feature extractor and training a separate classifier on top of its extracted features.
 
3. Interpretability and Explainability: 
- Grad-CAM: Gradient-weighted Class Activation Mapping, a technique to visualize the regions of an image that contribute most to the model's prediction. 
- SHAP (Shapley Additive Explanations): An approach to explain the predictions of complex models by assigning importance values to the input features.
 
4. Generative Adversarial Networks (GANs): 
- Generator: A network that generates new samples by mapping random input vectors to the desired data distribution. 
- Discriminator: A network that distinguishes between real and generated samples, providing feedback to the generator for training. 

5. Self-Supervised Learning: 
- Pretext Task: A proxy task designed to learn useful representations from unlabeled data, which can later be fine-tuned for downstream tasks. 
- Contrastive Learning: A self-supervised learning approach that trains the model to differentiate positive pairs from negative pairs in the latent space.
 
6. Autoencoders: 
- Encoder: A network that compresses the input data into a low-dimensional representation. 
- Decoder: A network that reconstructs the input data from the encoded representation, aiming to minimize the reconstruction error.

#### These terms encompass various aspects of working with convolutional and deep neural networks, including their optimization, upgrade, training, testing, and validation processes. 

7. Optimization:
- Gradient Descent: An iterative optimization algorithm that updates the network's parameters based on the gradients of the loss function with respect to the parameters.
- Learning Rate: The scalar value that determines the step size at each iteration during gradient descent. It controls the speed of convergence.
- Backpropagation: The process of computing the gradients of the loss function with respect to the network parameters by propagating the error backwards through the network.

8. Upgrading:
- Transfer Learning: Utilizing pre-trained models on large-scale datasets and adapting them to a new task or dataset by fine-tuning or feature extraction.
- Model Architecture Modification: Modifying the structure of the network, such as adding or removing layers, changing layer sizes, or introducing new modules to improve performance.

9. Training:
- Epoch: One complete pass of the entire training dataset through the network during the training process.
- Loss Function: A function that measures the discrepancy between the predicted output and the ground truth labels, providing a measure of the network's performance.
- Mini-batch: Dividing the training dataset into smaller subsets called mini-batches, which are processed sequentially during training, allowing for more efficient computation and utilization of parallel processing.

10. Testing:
- Test Dataset: An independent dataset used to evaluate the performance of the trained network after training.
- Accuracy: The measure of how well the network's predictions match the ground truth labels in the test dataset.
- Confusion Matrix: A matrix that summarizes the predictions made by the network, providing insight into true positives, true negatives, false positives, and false negatives.

11. Validation:
- Validation Dataset: A separate dataset used to assess the performance of the network during the training process, providing an estimate of generalization performance.
- Early Stopping: A technique where training is halted if the performance on the validation dataset does not improve for a certain number of epochs, preventing overfitting.

12. Hyperparameter Tuning:
- Learning Rate: The rate at which the network adjusts its parameters during training.
- Batch Size: The number of training examples processed in one iteration before updating the network's parameters.
- Regularization: Techniques like L1 or L2 regularization to prevent overfitting by adding penalty terms to the loss function.

#### Understanding these terms will help you navigate and optimize convolutional and deep neural networks, ensuring effective training, evaluation, and deployment of models for various tasks.

1. Regularization:
- Dropout: A regularization technique that randomly sets a fraction of the neurons to zero during training to prevent overfitting.
- Batch Normalization: Normalizing the input to each layer of the network by subtracting the batch mean and dividing by the batch standard deviation, which helps stabilize and speed up training.

2. Loss Functions:
- Mean Squared Error (MSE): Calculates the average squared difference between predicted and true values.
- Cross-Entropy Loss: Measures the dissimilarity between the predicted probability distribution and the true distribution.

3. Activation Functions:
- ReLU (Rectified Linear Unit): An activation function that returns the maximum of zero or the input value.
- Sigmoid: Maps the input to a value between 0 and 1, often used for binary classification tasks.
- Softmax: Converts a vector of real numbers into a probability distribution, commonly used for multi-class classification.

4. Model Evaluation Metrics:
- Precision: The ratio of true positives to the sum of true positives and false positives, measuring the accuracy of positive predictions.
- Recall: The ratio of true positives to the sum of true positives and false negatives, measuring the ability to identify positive instances.
- F1 Score: The harmonic mean of precision and recall, providing a balanced measure of model performance.

5. Optimization Techniques:
- Adam: An optimization algorithm that combines adaptive learning rates and momentum.
- Stochastic Gradient Descent (SGD): A variant of gradient descent that randomly selects a subset of training samples (mini-batch) for each iteration.
- Learning Rate Decay: Decreasing the learning rate over time to fine-tune the model as training progresses.

6.Data Augmentation:
- Image Rotation: Rotating the image by a certain angle to create additional training examples.
- Image Flipping: Horizontally or vertically flipping the image to increase the diversity of the training data.

7. Overfitting and Underfitting                    
- Overfitting: When a model performs well on the training data but fails to generalize to unseen data due to memorizing the training examples.
- Underfitting: When a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test sets.

### Terms and Definition 
1. Conv2D 
2. Dropout 
3. Maxpooling2D 
4. Activation Function 
5. Kernel Initializer 
6. Padding 
7. Conv2DTranspose 
8. Concatenate 
9. Upsampling2D 
10. Downsampling 
11. Hyperparameter Tuning 
12. Learning parameter 
13. Bias 
14. Variance 
15. Confusion Matrix 
16. Accuracy 
17. Precision 
18. Recall 
19. F1score 
20. Optimizer 
21. Loss/Cost/Error Funcyion 
22. Metrics 
23. Model Check point 
24. Early stopping 
25. Tensor Board 
26. Validation split 
27. Batch_size 
28. Epochs 
29. Callbacks 
30. Verbose 
31. Image - Channel, Height and Width 
32. Batch Normalization 
33. Lamda Keras layer 
34. Mean Average Presicion (mAP) 
35. Weights 
36.
