
### Choosing a Deep Learning Framework: Tensorflow or Pytorch?

1. PyTorch:
- PyTorch is one of the newest deep learning framework which is gaining popularity due to its simplicity and ease of use. Pytorch got very popular for its dynamic computational graph and efficient memory usage. Dynamic graph is very suitable for certain use-cases like working with text.
- Back-propogation uses the dynamically built graph
- Pytorch is easy to learn and easy to code. For the lovers of oop programming, torch.nn.Module allows for creating reusable code which is very developer friendly. Pytorch is great for rapid prototyping especially for small-scale or academic projects.

2. Caffe2:
- Another framework supported by Facebook, built on the original Caffe was actually designed by Caffe creator Yangqing Jia. It was designed with expression, speed, and modularity in mind especially for production deployment which was never the goal for Pytorch.
- Recently, Caffe2 has been merged with Pytorch in order to provide production deployment capabilities to Pytorch but we have to wait and watch how this pans out. Pytorch 1.0 roadmap talks about production deployment support using Caffe2.

3. Caffe:
- Caffe is a Python deep learning library developed by Yangqing Jia at the University of Berkeley for supervised computer vision problems. It used to be the most popular deep learning library in use. Written in C++, Caffe is one of the oldest and widely supported libraries for CNNs and computer vision. Nvidia Jetson platform for embedded computing has deep support for Caffe(They have added the support for other frameworks like Tensorflow but it’s still not enough). The same goes for OpenCV, the widely used computer vision library which started adding support for Deep Learning models starting with Caffe. For years, OpenCV has been the most popular way to add computer vision capabilities to mobile devices. So, if you have a mobile app which runs openCV and you now want to deploy a Neural network based model, Caffe would be very convenient.

4. CNTK:
- Microsoft Cognitive toolkit (CNTK) framework is maintained and supported by Microsoft. Since we have limited experience with CNTK, we are just mentioning it here. However, it’s not hugely popular like Tensorflow/Pytorch/Caffe.

5. Tensorflow:
- Tensorflow, an open source Machine Learning library by Google is the most popular AI library at the moment based on the number of stars on GitHub and stack-overflow activity. It draws its popularity from its distributed training support, scalable production deployment options and support for various devices like Android.
- One of the most awesome and useful thing in Tensorflow is Tensorboard visualization. - In general, during train, one has to have multiple runs to tune the hyperparameters or identify any potential data issues. Using Tensorboard makes it very easy to visualize and spot problems.
- Tensorflow Serving is another reason why Tensorflow is an absolute darling of the industry. This specialized grpc server is the same infrastructure that Google uses to deploy its models in production so it’s robust and tested for scale. In Tensorflow Serving, the models can be hot-swapped without bringing the service down which can be crucial reason for many business.
- In Tensorflow, the graph is static and you need to define the graph before running your model. Although, Tensorflow also introduced Eager execution to add the dynamic graph capability.
- In Tensorflow, entire graph(with parameters) can be saved as a protocol buffer which can then be deployed to non-pythonic infrastructure like Java which again makes it borderless and easy to deploy.

6. MXNet:
- Promoted by Amazon, MxNet is also supported by Apache foundation. It’s very popular among R community although it has API for multiple languages. It’s also supported by Keras as one of the back-ends.

7. Torch:
- Torch (also called Torch7) is a Lua based deep learning framework developed by Clement Farabet, Ronan Collobert and Koray Kavukcuoglu for research and development into deep learning algorithms. Torch has been used and has been further developed by the Facebook AI lab. However, most of force behind torch has moved to Pytorch.

8. Theano:
- Theano was a Python framework developed at the University of Montreal and run by Yoshua Bengio for research and development into state of the art deep learning algorithms. It used to be one of the most popular deep learning libraries. The official support of Theano ceased in 2017.

9. DeepLearning4j:
- DeepLearning4J is another deep Learning framework developed in Java by Adam Gibson.
“DL4J is a JVM-based, industry-focused, commercially supported, distributed deep-learning framework intended to solve problems involving massive amounts of data in a reasonable amount of time.”

- As you can see, that almost every large technology company has its own framework. In fact, almost every year a new framework has risen to a new height, leading to a lot of pain and re-skilling required for deep learning practitioners.
- The world of Deep Learning is very fragmented and evolving very fast. Look at this tweet by Karpathy:
Imagine the pain all of us have been enduring, of learning a new framework every year.

10. Keras:
- François Chollet, who works at Google developed Keras as a wrapper on top of Theano for quick prototyping. Later this was expanded for multiple frameworks such as Tensorflow, MXNet, CNTK etc as back-end.

- Keras is being hailed as the future of building neural networks. Here are some of the reasons for its popularity:
- Light-weight and quick: Keras is designed to remove boilerplate code. Few lines of keras code will achieve so much more than native Tensorflow code. You can easily design both CNN and RNNs and can run them on either GPU or CPU.
- Emerging possible winner: Keras is an API which runs on top of a back-end. This back-end could be either Tensorflow or Theano. Microsoft is also working to provide CNTK as a back-end to Keras.

- Currently, Keras is one of the fastest growing libraries for deep learning. The power of being able to run the same code with different back-end is a great reason for choosing Keras. Imagine, you read a paper which seems to be doing something so interesting that you want to try with your own dataset. Let’s say you work with Tensorflow and don’t know much about Torch, then you will have to implement the paper in Tensorflow, which obviously will take longer. Now, If the code is written in Keras all you have to do is change the back-end to Tensorflow. This will turbocharge collaborations for the whole community.
