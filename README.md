# Generative-Adveserial-Networks-for-Synthetic DNA

**Generative Adversarial Networks to increase DNA data size and discover RNA recognition patterns of CTCF**
**Sahithi Sunkishala**
Dr.Soibam Benjamin, Research Mentor, DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING TECHNOLOGY, UHD.

**Abstract:**

A DNA motif is defined as a nucleic acid sequence pattern that has some biological significance such as being DNA binding sites for a regulatory protein. Deep learning models have been successfully applied to extract DNA sequence motifs and regulatory grammars in the context of transcription factor binding. The goal of this project is to identify chromatin motifs representing short and recurring patterns exhibited by 12 epigenetic modifications, CTCF, and RAD21 binding signals in  boundaries of topologically associated boundaries. RNA is a critical component of chromatin in eukaryotes. We have used previously developed DeepLncCTCF, a new deep learning model based on a convolutional neural network and a bidirectional long short-term memory network, to discover the RNA recognition patterns of CTCF and identify candidate lncRNAs binding to CTCF and evaluated on human U2OS dataset. 

We proposed several generative neural network methods to generate DNA sequence data. We present 3 approaches and compare performance of these on the DeepLncCTCF model that identifies RNA motifs in genome. Generative Adversarial Networks are deep learning architecture generative models that have seen wide success in generating real like synthetic images with good accuracies. Aim of this project is to evaluate the models performance by increasing the train data size using 3 approaches of Generative adversarial methods such as conditional-GAN, AC-GAN, FC-GAN and comparing the results of GAN models to see which GAN model improves the model performance. Generative adversarial networks (GANs), conditional GANs attempt to better direct the data generation process by conditioning. AC-GAN has a discriminator that predicts the labels. FC – GAN discriminator has an advanced auxiliary classifier which distinguishes each real class from an extra 'fake' class. These GAN models are designed to generate artificial sequences to increase training dataset. Artificial data is generated with above GAN models and DeepLncCTCF is trained with these new datasets. Model are compared on metrics AUC, accuracy, sensitivity, specificity and MCC. It is anticipated that the proposed methods have a certain effect on DeepLncCTCF performance.

**Generator**
We have used same generator model for all the 3 types but used different Discriminator architectures.

**Discriminator** 
CGAN - For CGAN, the inputs to the discriminator are DNA sequence data and its labels. The output is the probability that the image is real or fake.


AC_CGAN - the input to the discriminator is a DNA sequence data and its labels, whilst the output is the probability that the image is real and its class label.

FC_CGAN – the discriminator has an advanced auxiliary classifier which distinguishes each real class from an extra ‘fake’ class. The ‘fake’ class avoids mixing generated data with real data, which can potentially confuse the classification of real data as AC-GAN does, and makes the advanced auxiliary classifier behave as another real/fake classifier.
<img width="628" alt="image" src="https://user-images.githubusercontent.com/102439554/163279641-30ed3414-86c6-4798-8b7d-4b75b04e25a6.png">


<img width="850" alt="image" src="https://user-images.githubusercontent.com/102439554/163270182-2c38d965-41a5-46bd-ae35-a7ae5ef9c8b5.png">
bestmodels link: https://uhdowntown-my.sharepoint.com/:f:/g/personal/sunkishalas1_gator_uhd_edu/EvnzIEVw0BVKh_szydulUf4BuNtDoHqQ9w0MZMOJoDLuFg?e=wIiaky

**Conclusion:**

We were able to increase the accuracy of the model to 79% from 77% using CGAN model. This tells us that the 30000 fake data generated is not entirely useful in improving the model accuracy. This might be due to the class labels not classified correctly as Conditional GAN doesn’t generate labels data.  Hence we have decided to use GAN methods which can generate class labels for fake data such as Auxiliary Classifier GAN. 


The auxiliary classifier assigns each real sample to its specific class and each generated sample to the class corresponding to the generator input. From the real vs fake loss plot we can see that the AC GAN model is not saturating after a point and it is converging. Also from the class label real vs fake plot we can see that assigning fake DNA data with their class labels the same way as real data is confusing the auxiliary classifier.

To overcome this we have implemented Fast convergence GAN that introduces an advanced auxiliary classifier for the purposes of fast convergence and improved quality. FC-GAN has proved to generate images and classify images correctly to real or fake labels but FC-GAN model discriminator needs improvement as it is not able to classify the class labels as expected for DNA sequence data and there is no improvement in accuracy.



References: https://academic.oup.com/nargab/article/2/2/lqaa031/5831011#209720175

https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/#:~:text=The%20Auxiliary%20Classifier%20GAN%2C%20or,than%20receive%20it%20as%20input

https://arxiv.org/abs/1805.01972




