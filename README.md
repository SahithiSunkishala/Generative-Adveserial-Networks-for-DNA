# Generative-Adveserial-Networks-for-DNA

Generative Adversarial Networks to increase DNA data size and discover RNA recognition patterns of CTCF
Sahithi Sunkishala
Dr.Soibam Benjamin, Research Mentor, DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING TECHNOLOGY, UHD.

Abstract:
A DNA motif is defined as a nucleic acid sequence pattern that has some biological significance such as being DNA binding sites for a regulatory protein. Deep learning models have been successfully applied to extract DNA sequence motifs and regulatory grammars in the context of transcription factor binding. The goal of this project is to identify chromatin motifs representing short and recurring patterns exhibited by 12 epigenetic modifications, CTCF, and RAD21 binding signals in  boundaries of topologically associated boundaries. RNA is a critical component of chromatin in eukaryotes. We have used previously developed DeepLncCTCF, a new deep learning model based on a convolutional neural network and a bidirectional long short-term memory network, to discover the RNA recognition patterns of CTCF and identify candidate lncRNAs binding to CTCF and evaluated on human U2OS dataset. 

We proposed several generative neural network methods to generate DNA sequence data. We present 3 approaches and compare performance of these on the DeepLncCTCF model that identifies RNA motifs in genome. Generative Adversarial Networks are deep learning architecture generative models that have seen wide success in generating real like synthetic images with good accuracies. Aim of this project is to evaluate the models performance by increasing the train data size using 5 approaches of Generative adversarial methods such as conditional-GAN, AC-GAN, FC-GAN and comparing the results of GAN models to see which GAN model improves the model performance. Generative adversarial networks (GANs), conditional GANs attempt to better direct the data generation process by conditioning. AC-GAN has a discriminator that predicts the labels. FC – GAN discriminator has an advanced auxiliary classifier which distinguishes each real class from an extra 'fake' class. These GAN models are designed to generate artificial sequences to increase training dataset. Artificial data is generated with all GAN models and DeepLncCTCF is trained with all these datasets. Model are compared on metrics AUC, accuracy, sensitivity, specificity and MCC. It is anticipated that the proposed methods have a certain effect on DeepLncCTCF performance.

Generator:

We have used same generator model for all the 3 types but used different Discriminator architectures.

Discriminator:
CGAN - For CGAN, the inputs to the discriminator are DNA sequence data and its labels. The output is the probability that the image is real or fake.





AC - CGAN - the input to the discriminator is a DNA sequence data and its labels, whilst the output is the probability that the image is real and its class label.

FC-CGAN – the discriminator has an advanced auxiliary classifier which distinguishes each real class from an extra ‘fake’ class. The ‘fake’ class avoids mixing generated data with real data, which can potentially confuse the classification of real data as AC-GAN does, and makes the advanced auxiliary classifier behave as another real/fake classifier.


![Uploading image.png…]()







References: https://academic.oup.com/nargab/article/2/2/lqaa031/5831011#209720175


