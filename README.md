<BODY>

<DIV id="id_1">
<H1> Evaluating Bag of Features in Scene Classification</H1>
</DIV>
<DIV id="id_2">
<P class="p7 ft4">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The demand for automation of machine in today’s world led to unsupervised learning. Bag of words being one such unsupervised technique, is dated only a decade back. However, research in this area has been no less. This project presents you with detailed implications of feature detection and extraction techniques on the Bag of Words’ application. In previous studies, detailed analysis of: dimension complexity, time complexity and space complexity has not been attempted. Also, we present to readers, different scenarios in which a particular method would be beneficial over the other.</P>
</DIV>
<DIV id="id_2_1">
<P class="p9 ft6"><H2> Introduction </H2></P>
<P class="p10 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scene Classification is one of the most advancing areas in the field of Image Processing. Because of its use in: weather forecast, browsing for a particular class of images, recognition of objects within a scene, medical diagnosis; demand for efficient scene classification algorithm rises. On one hand, if scene classification deals with images, on the other, text mining uses textual words. With a <NOBR>large–scale</NOBR> collection of data, efficient and robust classification techniques are needed for easy search. Scene classification of images depends mainly on: texture, transformation, color of the image. Use of efficient local feature detection and extraction techniques is necessary to capture these details.</P>
</DIV>
![sceneclassification](https://cloud.githubusercontent.com/assets/11435669/20867664/fda02e2e-ba17-11e6-8702-b0d0a35bd3e6.png)
<DIV id="id_2_2">
<P class="p12 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Researchers have come up with various feature detection and extraction algorithm. In this paper, we implement some prominent local feature extraction methods. Feature extraction is the retrieval of important information which describes the system, many a times better than the system itself. We believe each feature extraction technique suites better for its respective system, it is designed for. Therefore, an attempt has been made in this report to provide this correlation of system to extraction technique. Henceforth, we experiment with our features providing: dimensional and time analysis. Further, we show a real world application of feature extraction in scene classification.</P>
<P class="p13 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Bag of words a.k.a Bag of Visual Words as implied to images is an unsupervised classification method. Bag of Words clusters histogram of parts. The term dictionary in the Bag of Words is analogous to linguistic dictionary; in a way that it contains a histogram representation of image parts. And the term bag of words, symbolizes clustering these histograms into classes. Thereby, different classes represent the classification of images. This project presents, added dimensional complexity of scene classification using the Bag of Words; which most researchers have not dealt with. Our motivation for this project budded from <NOBR>Image–net</NOBR> scene classification challenge. Also, this project is our startling attempt to classify images with support of Dr. Dapeng Wu. We believe, this paper will benefit readers researching to compare various feature detection and extraction methods, unsupervised learning with bag of words, scene classification. We hope ample visual content helps in better understanding.</P>
</DIV>
<DIV id="id_2_3">
<P class="p14 ft6"><H2> Related Work </H2></P>
<P class="p15 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scene classification is a rigorously researched area. Vailaya et al. [1] is regarded as one of the forerunners to use global features to classify images. To mention few prominent research papers in the field of bag of words:</P>
<P class="p13 ft9">The first paper on scene classification using BOW by Lazebnik et al. [3] segregated image into grids. Then the spatial pyramid model was built upon taking histogram of these grids. Second paper on bag of words, by Monay et al. [4] used probabilistic approach. They built upon the probability of classification using BOW finding out the semantic difference between natural and <NOBR>man-made</NOBR> scenes. Third paper by Boutell et al<SPAN class="ft13">. </SPAN>[5] configured pair wise relationship between different scenes. Also paper by Gemert et al. [6] presented similar concept of occurring histogram.</P>
<P class="p17 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Comparisons of Local feature extraction and detection is found in abundance. Some to mention: Paper by Chao et al.<SPAN class="ft6">[2]</SPAN><SPAN class="ft14">compared various feature detectors and descriptor for image retrieval in </SPAN><NOBR>JPEG–encoded</NOBR> query images. Also, paper by Patel et al [7] compared various feature detectors and descriptor for real time face tracking. Additionally, stand- alone research papers on local feature detector and descriptors helped us in better understanding and applying the same in our project. In 2004 David G. Lowe introduced SIFT as descriptor as well as extractor. Further, as improvement to SIFT, Bay et al. [8] introduced SURF. Later in 2006 Rosten et al [9] introduced the FAST algorithm for real time applications.</P>
<P class="p19 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As we can see the comparison of detectors and descriptors for BOW has still a long way to go. Also, with observation, scene classification using BOW produces better results with local feature extraction than global. Therefore, in our report, we explore various local feature extraction techniques and aim to provide the best design specific to the system.</P>

<DIV id="id_2_4">
<P class="p20 ft6"><H2> Methods </H2></P>
<P class="p21 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The first step in scene classification is to detect and extract Features. We evaluate various feature detection and extraction algorithm for various databases of UICU sports database, MIT indoor database, <NOBR>scenes–15</NOBR> and Motajabi’s workshop database. Further, we explore matching techniques of Flann Based and Brute force using Hamming distance. With our input data as feature vectors, we implement unsupervised classification using Bag of Visual Words. Bag of words is evaluated for various combinations of feature extraction and detection algorithms. Use of machine learning classifier includes usage of <NOBR>K–means</NOBR> and SVM. <NOBR>K–means</NOBR> is an unsupervised classifier, whereas, SVM is a supervised classifier.</P>
</DIV>

<DIV id="id_1_2">
<P class="p22 ft10"><SPAN class="ft10"><H3> A.&nbsp;&nbsp; Feature Detection and Extraction </H3></P>
<P class="p23 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Feature detection is corner or blob detection in an image. These are called key region of interest, as they can be easily located even with transformation of images. Whereas, feature extractor computes <NOBR>feature–vectors</NOBR> on key region of interest. In our project we use known complex detection methods: SIFT, SURF, FAST, STAR, MSER, GFFT and DENSE and extraction methods: SIFT, SURF, BRIEF.</P>
</DIV>

![compare](https://cloud.githubusercontent.com/assets/11435669/20867665/fda21b94-ba17-11e6-972d-34351ddb50e3.png)
<P class="p24 ft9">Right from top: SIFT, SURF, BRIEF, FAST, Left from top: STAR, MSER, GFFT, DENSE Feature Extractors
</P>


<DIV id="id_1">
<P class="p22 ft10"><SPAN class="ft10"><H4> 1.&nbsp;&nbsp; Scale-invariant feature transform - SIFT</H4></P>
<P class="p34 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Three steps of binary detection are: Firstly, find sampling points called key regions or the special regions. Secondly, map key points in order to achieve rotational invariance. Lastly, Refine key points, discarding unimportant ones.</P>
<P class="p35 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sift feature detector is an improvement over the Harris corner detector, which detects corners even for rotated images and scaled images. Sift was introduced by David G. Lowe. There are three steps in SIFT feature detection. Firstly, scale– space is implemented using difference of Gaussian filters. Also, it is necessary to implement different window size for different scale of the image. Therefore, the window size is chosen based on image scale, and local extrema for the image is configured. Secondly, Key point localization technique is used to refine extrema by thresholding. Thirdly, each key point is made invariant to orientation by constructing an orientation map for each key point.</P>
<P class="p36 ft6">For feature extraction, 16 x 16 neighboring pixels of a key point are considered. Then, with 4 x 4 sub blocks, we have a total of 16 regions. Considering 8 bins of histogram for each region, gives us a total of 128 dimensions for each key point. Over 128 x n, with n being number of key points, rotational illumination invariance techniques are applied.</P>
</DIV>

<DIV id="id_2">
<P class="p37 ft10"><SPAN class="ft10"><H4> 2.&nbsp;&nbsp; Speeded up Robust features – SURF</H4></P>
<P class="p39 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SURF is the faster version of SIFT. However, our experiments show: SURF feature extractor is less efficient than SIFT. Compared to SIFT, SURF has added LOG approximation and BOX filtering. Using, BOX filters, with integral images, we can easily find convolution in parallel for all key points. In order to achieve scale invariance SURF uses Hessian matrix. Also, SURF uses Gaussian weights in a linear direction to achieve rotational invariance.</P>
<P class="p40 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For feature extraction, SURF uses linear wavelets. Around detected key point, 20 x 20 neighboring pixels are considered. Further, 20 x 20 regions are segregated into 4 x 4 sub regions. Applying vector <SPAN class="ft20">V = (sigma∑dx , ∑dy , ∑ </SPAN><SPAN class="ft21">|dy| </SPAN><SPAN class="ft20">, ∑ </SPAN><SPAN class="ft21">|dx|)</SPAN><SPAN class="ft20">, </SPAN>we can achieve 64 dimensional feature vector for each key point.</P>
</DIV>

<DIV id="id_3">
<P class="p38 ft10"><SPAN class="ft10"><H4> 3.&nbsp;&nbsp; Features from accelerated segment test - FAST</H4></P>
<P class="p42 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FAST feature detection algorithm emerged for real time high speed applications. It was proposed by Roster et al [9]. In FAST, thresholding selects key points. Example: for a pixel 'p', 16 pixels around it are considered in circular fashion. Then all the magnitude of pixel points in the circles are compared with 'p'. Then, 'p' is considered a key point only if it is the maximum or minimum of the set. Similarly corner test is performed considering 3 surrounding pixels.</P>
<P class="p43 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;There are certain limitations to FAST feature detection. Feature key points which can be found with no delay are ignored by FAST algorithm. Also, key points derived from the FAST technique are too close to each other. The latter limitation can be resolved by using maximal suppression. Over here, distance between key points is set at a constant threshold value. Even if, FAST feature detection excels over
HARRIS corner detection in detecting corners</P>
</DIV>

<DIV id="id_1_2">
<P class="p45 ft10"><SPAN class="ft10"><H4> 4.&nbsp;&nbsp; Maximally Stable Extremal Regions - MSER</H4></P>
<P class="p46 ft24">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MSER was introduced by Matas et al [10] which considered connection between images. MSER is known to work better for blurred images. Additionally, MSER achieves higher classification results for images with different illumination levels. As, MSER considered correspondence points, it outputs key regions instead of key points. Also, it is seen to it that stable regions are selected over unstable regions by thresholding. Multiscale detection of key regions by multi scale pyramid helps in achieving invariance to scale. Overall, MSER is one of the strongest feature detectors as it is blur, illumination and scale invariant.</P>
</DIV>

<DIV id="id_1_3">
<P class="p38 ft10"><SPAN class="ft10"><H4> 5.&nbsp;&nbsp; Oriented FAST and rotated BRIEF – ORB</H4></P>
<P class="p46 ft24">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ORB was first introduced by Rublee et al [11]. as an alternative to SIFT and SURF. ORB is basically a modified version of fussed of FAST and BRIEF. Steps involved in ORB feature detection: In the first step, it implements the FAST algorithm and Harris corner detector. The FAST algorithm finds key points. Further, rotational invariance is achieved by calculating the centroid of the key point. The direction pointing from the centroid of the key point to the corner; detected by Harris key point is the orientation map of the key point. Therefore, ORB implements FAST and provides orientation information.</P>
</DIV>

<DIV id="id_1_1">
<P class="p38 ft10"><SPAN class="ft10"><H4> 6.&nbsp;&nbsp; Good features to track – GFFT</H4></P>
<P class="p73 ft24">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GFFT was introduced by Shi and Thomas, to detect corners. For each pixel Gradient Matrix is applied. Here, integral values of the image are used. Henceforth, computationally it is constant. A maximum detects optimum features. However, with motion effects, there is a resulting error in generalization of the aperture. Even though thresholding and <NOBR>non–maximal</NOBR> suppression are used, GFFT performs poorly for images with motion effects.</P>
</DIV>

<DIV id="id_1_1">
<P class="p38 ft10"><SPAN class="ft10"><H4> 7.&nbsp;&nbsp; DENSE features</H4></P>
<P class="p73 ft24">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The DENSE feature detection provides good coverage of the entire image. Here key points are considered in step and pixels whose contrasts vary the most are considered. Stepwise implementation helps in extracting all important information of an image. Also, more information can be extracted with the overlap of the patches. However, all these increase dimensional complexity.</P>
</DIV>

<DIV id="id_1_1">
<P class="p38 ft10"><SPAN class="ft10"><H4> 8.&nbsp;&nbsp; Binary Robust Invariant Scalable Key points – BRISK</H4></P>
<P class="p76 ft24">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BRISK unlike ORB, has a handmade sampling pattern as shown in Fig. 4. Gaussian filter is applied to different regions separately. Additionally, threshold distinguishes patterns into short, long and unused pairs as shown in Fig. 5. Wherein, short hand distance pairs are used to compute intensity values, whereas long hand distance pairs are used to find orientation. Like ORB, BRISK performs better for view point change.</P>
</DIV>

<DIV id="id_1_2">
<P class="p38 ft10"><SPAN class="ft10"><H4> 9.&nbsp;&nbsp; STAR feature detection</H4></P>
<P class="p79 ft24">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Center surrounded extrema was introduced by Agarwal et al. STAR uses 2 rotated squares to extract <NOBR>bi–level</NOBR> features of center surrounded extrema. Also, it considers 7 scales of the image. Unlike other feature detection technique, sampling size is fixed. For each 7 scales, spatial resolution of the complete image is considered. Finally, it is followed by detection of edges by Gradient matrix and line suppression.
</P>
</DIV>

![table1](https://cloud.githubusercontent.com/assets/11435669/20868067/13308ff6-ba20-11e6-8933-bbb46c52260c.png)
![table2](https://cloud.githubusercontent.com/assets/11435669/20868066/13068148-ba20-11e6-97ca-37b8dbc30149.png)

<DIV id="id_1_3">
<P class="p80 ft10"><SPAN class="ft10"><H3> B.&nbsp;&nbsp; Feature Matching </H3></P>
<P class="p81 ft10"><SPAN class="ft10"><H4> 1.&nbsp;&nbsp; Brute Force Matcher – BF Matcher </H4></P>
<P class="p79 ft24">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Brute force matcher takes a descriptor of one feature of an image and iterates over the features of the other image till it finds the feature key point match. Here we use hamming distance. Brute force matcher increases time complexity as it iterates over complete image key points.
</P>
</DIV>

<DIV id="id_3_2">
<P class="p81 ft10"><SPAN class="ft10"><H4> 2.&nbsp;&nbsp; FLANN –Fast Library for Approximate Nearest Neighbors </H4></P>
<P class="p89 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Unlike BF Matcher, FLANN doesn’t iterate over all the features of the image, but using search trees; it derives approximate search. For high dimensional features and large dataset, FLANN comes with inbuilt fast nearest search of the key points. Comparative studies between BF Matcher and FLANN reveal BF Matcher more accurate than FLANN Matcher. However, FLANN based matcher is faster than BF Matcher even for large dimensional key points and large dataset. For the Bag of Visual Words clustering over a million features, in our project we stick to FLANN matcher.</P>
</DIV>

<DIV id="id_1">
<P class="p80 ft10"><SPAN class="ft10"><H3> C.&nbsp;&nbsp; Bag of Words </H3></P>
<P class="p90 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Bag of words is well known used method for classification of images and text. It originated from text mining of frequency of words in the dictionary. Opposite to supervised classification techniques, Bag of Words learns frequency of occurrence and learns the pattern on its own and generates code book. This code book is nothing but the histogram representation of each part as shown in Fig. 5. Codebook is analogous to words in the dictionary. Each word is a cluster of similar histogram patterns. Let discuss internal mathematical implementations in the Bag of Words.</P>
<P class="p91 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Bag of Words is also used in object recognition analogous to scene classification. Internally, it evaluates using confusion matrix. Bayes model developed in Natural Language Processing <NOBR>–NLP</NOBR> can be used even for image classification.</P>
<P class="p92 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Let ‘<SPAN class="ft20">c’ </SPAN>be image category, ‘<SPAN class="ft20">V’ </SPAN>be the size of Codebook and ‘<SPAN class="ft20">w’ </SPAN>be <SPAN class="ft20">V </SPAN><NOBR><SPAN class="ft20">–dimensional</SPAN></NOBR><SPAN class="ft20"> </SPAN>vector. In matrix of ‘<SPAN class="ft20">w’ </SPAN>only one entity is 1.</P>
<P class="p93 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;W = [w<SPAN class="ft39">1 </SPAN>, w<SPAN class="ft39">2 </SPAN>,………………, w<SPAN class="ft39">n</SPAN>]</P>
<P class="p94 ft6"><SPAN class="ft10">W</SPAN><SPAN class="ft40">is the representation of the image using all the patches </SPAN><SPAN class="ft10">‘w’ </SPAN>in the image.</P>
<P class="p95 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Naïve Bayes classifier is used in categorization. Bag of Words classifier has to learn each categorization for an image. For example: nose, mouth for a face and wheel, break for a bike as shown in Fig. 5. This particular object level categorization is possible by using the Naïve Bayes classifier. 
<P class="p97 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The higher levels of implementations of Bag of Words are VLDA encoding and Spatial Pyramid Pooling. Spatial pyramid pooling uses various resolutions of feature level extraction. Using Spatial Pyramid pooling, M. Koskela achieved the highest <NOBR>Image–net</NOBR> scene classification accuracy of around 91%. This is by using combinational advantages of <NOBR>Convolutional–Neural–Network–CNN</NOBR> and Spatial Pyramid Pooling. It took about 13 days of training on each set.</P>
<P class="p98 ft8">Fig. 6.Bag of Words.</P>
</DIV>

<DIV id="id_1_2">
<P class="p80 ft10"><SPAN class="ft10"><H3> D.&nbsp;&nbsp; Classifier </H3></P>
<P class="p99 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Different rules of machine learning classifiers are governed by different mathematical principles. First let’s concentrate on unsupervised machine learning classifier: <NOBR>K–means.</NOBR> <NOBR>K–means</NOBR> classifies <SPAN class="ft20">n </SPAN>observed classification into <SPAN class="ft20">n </SPAN>clusters. These clusters can have different shape, although two points in the cluster are at a comparable distance. Points are randomly chosen, these are called ‘Code Vector’. Then, points nearby are clustered when they satisfy minimum Euclidean distance.</P>
<P class="p100 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<NOBR>K–means</NOBR> is a heuristic algorithm of time complexity of <SPAN class="ft48">2</SPAN><SPAN class="ft49">Ω(</SPAN><SPAN class="ft50">n</SPAN><SPAN class="ft49">) </SPAN><SPAN class="ft51">. </SPAN>So, you cannot be sure that it will converge. Due to its time complexity, it takes squared more time for large data and would not suit the best.</P>
<P class="p102 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Now let’s consider supervised machine learning technique. There are two steps: first training of data and second is the prediction of class category. Most used supervised Machine Learning classifiers are KNN and SVM. Accuracy wise SVM performs better, however, speed wise KNN. For given K an integer KNN classifies based on only K neighbors. Whereas, SVM considers the whole data and tries to separate it into as many classes as there are scene categories. SVM separates data using hyperplanes. But, not all sets are linearly separable. Henceforth, there are other functions of SVM for higher dimensional space. These are called as kernel functions. Some to name: Quadratic function, Gaussian Radial Basis function– RBF. Considering recent evaluations of various SVM kernel functions, it was found that SVM with RBF function classifies the best. However, with the infinite dimension generalization error of RBF kernel was said to increase. When a dataset is small compared to the feature vector dimension, it results in a case similar to infinite dimension. Therefore, in our experiment we use Linear kernel function of SVM.</P>
</DIV>

<DIV id="id_1_2">
<P class="p80 ft10"><SPAN class="ft10"><H3> E.&nbsp;&nbsp; Cross Validation </H3></P>
<P class="p83 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cross validation sees to it that, no data are trained and tested at once. In this project threefold cross validation is used to find the overall accuracy of the set. The data are divided into three parts. 2/3<SPAN class="ft53">rd </SPAN>part is used for training and rest is used for testing. Likewise, it is repeated three times covering all the data for both testing and training. After deriving accuracy for each set, overall accuracy is found by averaging.</P>
</DIV>


<DIV id="id_1">
<P class="p80 ft10"><H2> EXPERIMENT </H2></P>
<P class="p104 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Evaluation of data included creation of 5 different datasets, with varied levels of complexity. <NOBR>Dataset–1</NOBR> rates to the easiest level of classification problem, while <NOBR>Dataset–5</NOBR> the most complex level.</P>
</DIV>

<DIV id="id_1_2">
<P class="p80 ft10"><SPAN class="ft10"><H3> A.&nbsp;&nbsp; Dataset </H3></P>
<P class="p104 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Testing of bag of words included creation of personal dataset. This data was created using a video sequence of <NOBR>Image–Net</NOBR> dataset. Videos at different time frames were captured. This included a class of bird and car as shown in Fig. consisting of 30 images each. Let this be <NOBR>dataset–1.</NOBR> While experimentation, we will refer this as <NOBR>Dataset–1.</NOBR></P>

![data-1](https://cloud.githubusercontent.com/assets/11435669/20868061/0bbdac2c-ba20-11e6-931a-1f64008554c9.png)

<P class="p106 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For <NOBR>Dataset–2</NOBR> of Motajabi’s workshop [12], we introduced a higher level of complexity. In total it included four classes of images, namely: airplane, bike, car and cheetahs. Each class has around 60 images each, so size of the dataset is 240. Variations ranged from the object being replaced with varied background scene to many objects in the scene. As seen in Fig. we trained and tested for many to one data of cheetahs.</P>

![data-2](https://cloud.githubusercontent.com/assets/11435669/20868058/0ba9ce14-ba20-11e6-9d6e-909e9ed6a996.png)

<P class="p108 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<NOBR>Dataset–3</NOBR> included elements like many objects, blur, far and close scenes, different image dimension. It was taken from the UIUC sports dataset [13], which included a total of 8 sports: rowing, badminton, polo, bocce, snowboarding, croquet, sailing, rock climbing. We selected 6 sports of total 8 and arranged them to find accuracy. Arranging of data is necessary in unsupervised learning to verify classification results of test data, if test data have no reference provided by Database provider. Therefore, our set consists of 6 sets of 120 images each.</P>

![data-3](https://cloud.githubusercontent.com/assets/11435669/20868059/0bb3ac7c-ba20-11e6-8715-bf9208c70e1c.png)

<P class="p110 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<NOBR>Dataset–4</NOBR> by MIT [14] has one of the highest levels of complexity. We arranged data into 10 classes of, namely: <NOBR>Airport–Inside,</NOBR> Bar, Bowling, Casino, <NOBR>Elevator–google,</NOBR> <NOBR>Inside–subway,</NOBR> <NOBR>locker–room,</NOBR> restaurant kitchen, and warehouse. Images of same classes sometimes have very little similarity. Look at Fig. 10. The top row shows restaurant kitchen. For a computer, matching images of almost no similarity possible only by using neural networks. The arranged data has a total of 10 classes of 90 images each.</P>

![data-4](https://cloud.githubusercontent.com/assets/11435669/20868060/0bbca606-ba20-11e6-88b3-a39f76b7ebbe.png)

<P class="p112 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dataset– 5 included <NOBR>scenes–15</NOBR> [15] of over seven million images. We used NVIDIA server with GPUs to run our algorithm. It took hours to run for KNN, but SVM would take days. Results show BOW to be inefficient for large data. Therefore, results are not evaluated for this dataset. However using CNN classification accuracy of 70 % was achieved.</P>
</DIV>

<DIV id="id_1">
<P class="p0 ft10">B. Evaluation</P>
<P class="p113 ft9">First step in our experiment, we evaluate feature detectors and extractors. Feature detectors such as SIFT, BRISK, ORB, SURF, FAST, STAR, MSER, GFFT and DENSE are fused with Feature extractors of SIFT and SURF. In our experiment, time taken for execution of different fusion techniques is compared. Feature extractors of SURF and SIFT produce almost the same results. As shown in Fig. 12. Taking normalized time along the positive <SPAN class="ft20">y </SPAN>axis, showed SIFT, GFFT and MSER to be efficient in time. However, this alone is not sufficient to evaluate time complexity of Bag of Words application.</P>
<!--[if lte IE 7]><P style="margin-left:27px;margin-top:0px;margin-right:-167px;margin-bottom:0px;" class="p114 ft55"><![endif]--><!--[if gte IE 8]><P style="margin-left:-113px;margin-top:0px;margin-right:-27px;margin-bottom:140px;" class="p114 ft55"><![endif]--><![if ! IE]><P style="margin-left:-43px;margin-top:70px;margin-right:-97px;margin-bottom:70px;" class="p114 ft55"><![endif]>time</P>
<P class="p115 ft8">Fig. 12. Time taken for Feature detection.</P>
<P class="p116 ft6">IV. RESULTS</P>
<P class="p117 ft9"><NOBR>Database–1</NOBR> has images with slight variation. Therefore, using Bag of Words we can achieve an accuracy of 100%. Number of features clustered by <NOBR>K–means</NOBR> has to be considered in order to evaluate space dimensionality. Fig. 13 shows DENSE features to have the highest number of features clustered. Here as the dimension is normalized, in our experiments: DENSE has a dimension of around 600,000 and FAST around 200,000, rest in the range of <NOBR>10–50</NOBR> hundreds. A low computing system fails to evaluate DENSE. Thereby, we discard DENSE in our further experiments.</P>
</DIV>
<DIV id="id_1_2">
<P class="p118 ft9">We once again fuse various feature detectors with SIFT and SURF feature extractors, for our <NOBR>Dataset–2.</NOBR> After analysis, SIFT feature extractor performed better than SURF feature extractor. However, with space complexity, few algorithms like GFFT cannot perform with SIFT. Therefore, even though less efficient, systems with space complexity works well with SURF. Otherwise, SIFT feature extractor produces better results. This comparison can be seen from Fig. 14 (top), wherein, accuracy obtained from SIFT and SURF are normalized. Also, from Fig. 14 (bottom) normalized error rates and dimensions are compared for different feature detectors with SIFT feature extractor. The lesser the dimension and error rates, the better the performance. Therefore, the FAST feature detector doesn’t seem to be perform better and can be ignored in our next experiment.</P>
</DIV>
</DIV>
<DIV id="id_2">
<P class="p119 ft8">Fig. 14. Comparison between SIFT and SIRF (top) , Dimension vs</P>
<P class="p120 ft8">Error rates of Feature Detectors with SIFT Feature Extractor.</P>
<P class="p121 ft6">For <NOBR>Dataset–3</NOBR> and <NOBR>Dataset–4,</NOBR> results are specific to the given system. These are given by table Table. 3. (top) and Table. 3. (bottom) respectively. First of all, for <NOBR>Dataset–3</NOBR> we can see that: even though GFFT is faster, it is less efficient, due to motion effects in sports activities. Also, it can be seen that <NOBR>SURF–SIFT</NOBR> has time complexity as the data is large. However, for <NOBR>Dataset–3</NOBR> <NOBR>SURF–SURF</NOBR> is the most efficient.</P>
<P class="p122 ft8">Fig. 13 Comparing dimensions of Feature detectors.</P>
<P class="p123 ft12">7</P>
</DIV>
</DIV>
<DIV id="page_8">


<DIV id="id_1">
<DIV id="id_1_1">
<P class="p124 ft9">Now, comparing results of <NOBR>Dataset–3</NOBR> to <NOBR>Dataset–4,</NOBR> it is evident that due to a smaller set of data in <NOBR>Dataset–4,</NOBR> SURF– SIFT is the most time efficient. Also, due to lesser motion effects, as <NOBR>Dataset–4</NOBR> is mostly composed of still objects, GFFT is the fastest with optimum efficiency.</P>
<TABLE cellpadding=0 cellspacing=0 class="t2">
<TR>
	<TD class="tr3 td22"><P class="p125 ft25"><NOBR>Detection–Extraction</NOBR></P></TD>
	<TD class="tr3 td23"><P class="p126 ft25">Accuracy</P></TD>
	<TD class="tr3 td24"><P class="p127 ft25">Time(minute)</P></TD>
</TR>
<TR>
	<TD class="tr12 td25"><P class="p50 ft56">&nbsp;</P></TD>
	<TD class="tr12 td26"><P class="p50 ft56">&nbsp;</P></TD>
	<TD class="tr12 td27"><P class="p50 ft56">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr0 td28"><P class="p69 ft8"><NOBR>SURF–SURF</NOBR></P></TD>
	<TD class="tr0 td29"><P class="p55 ft8">64</P></TD>
	<TD class="tr0 td30"><P class="p55 ft8">24</P></TD>
</TR>
<TR>
	<TD class="tr8 td31"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td32"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td33"><P class="p50 ft31">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr3 td28"><P class="p69 ft8"><NOBR>SURF–SIFT</NOBR></P></TD>
	<TD class="tr3 td29"><P class="p55 ft8">N/A</P></TD>
	<TD class="tr3 td30"><P class="p55 ft27">&gt;60</P></TD>
</TR>
<TR>
	<TD class="tr12 td31"><P class="p50 ft56">&nbsp;</P></TD>
	<TD class="tr12 td32"><P class="p50 ft56">&nbsp;</P></TD>
	<TD class="tr12 td33"><P class="p50 ft56">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr3 td28"><P class="p69 ft27"><NOBR>SIFT–SIFT</NOBR></P></TD>
	<TD class="tr3 td29"><P class="p55 ft8">62</P></TD>
	<TD class="tr3 td30"><P class="p55 ft8">30</P></TD>
</TR>
<TR>
	<TD class="tr12 td31"><P class="p50 ft56">&nbsp;</P></TD>
	<TD class="tr12 td32"><P class="p50 ft56">&nbsp;</P></TD>
	<TD class="tr12 td33"><P class="p50 ft56">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr3 td28"><P class="p128 ft8"><NOBR>BRISK–SIFT</NOBR></P></TD>
	<TD class="tr3 td29"><P class="p55 ft8">62</P></TD>
	<TD class="tr3 td30"><P class="p55 ft8">32</P></TD>
</TR>
<TR>
	<TD class="tr8 td31"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td32"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td33"><P class="p50 ft31">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr3 td28"><P class="p54 ft27"><NOBR>GFFT–SIFT</NOBR></P></TD>
	<TD class="tr3 td29"><P class="p55 ft8">48</P></TD>
	<TD class="tr3 td30"><P class="p55 ft8">20</P></TD>
</TR>
<TR>
	<TD class="tr12 td31"><P class="p50 ft56">&nbsp;</P></TD>
	<TD class="tr12 td32"><P class="p50 ft56">&nbsp;</P></TD>
	<TD class="tr12 td33"><P class="p50 ft56">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr4 td28"><P class="p125 ft25"><NOBR>Detection–Extraction</NOBR></P></TD>
	<TD class="tr4 td29"><P class="p126 ft25">Accuracy</P></TD>
	<TD class="tr4 td30"><P class="p127 ft25">Time(minute)</P></TD>
</TR>
<TR>
	<TD class="tr8 td25"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td26"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td27"><P class="p50 ft31">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr0 td28"><P class="p69 ft8"><NOBR>SURF–SURF</NOBR></P></TD>
	<TD class="tr0 td29"><P class="p55 ft8">41</P></TD>
	<TD class="tr0 td30"><P class="p55 ft8">23</P></TD>
</TR>
<TR>
	<TD class="tr8 td31"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td32"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td33"><P class="p50 ft31">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr0 td28"><P class="p69 ft8"><NOBR>SURF–SIFT</NOBR></P></TD>
	<TD class="tr0 td29"><P class="p55 ft8">46</P></TD>
	<TD class="tr0 td30"><P class="p55 ft8">30</P></TD>
</TR>
<TR>
	<TD class="tr8 td31"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td32"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td33"><P class="p50 ft31">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr3 td28"><P class="p69 ft27"><NOBR>SIFT–SIFT</NOBR></P></TD>
	<TD class="tr3 td29"><P class="p55 ft8">40</P></TD>
	<TD class="tr3 td30"><P class="p55 ft8">31</P></TD>
</TR>
<TR>
	<TD class="tr8 td31"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td32"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td33"><P class="p50 ft31">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr0 td28"><P class="p128 ft8"><NOBR>BRISK–SIFT</NOBR></P></TD>
	<TD class="tr0 td29"><P class="p55 ft8">42</P></TD>
	<TD class="tr0 td30"><P class="p55 ft8">33</P></TD>
</TR>
<TR>
	<TD class="tr8 td31"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td32"><P class="p50 ft31">&nbsp;</P></TD>
	<TD class="tr8 td33"><P class="p50 ft31">&nbsp;</P></TD>
</TR>
<TR>
	<TD class="tr3 td28"><P class="p54 ft27"><NOBR>GFFT–SIFT</NOBR></P></TD>
	<TD class="tr3 td29"><P class="p55 ft8">43</P></TD>
	<TD class="tr3 td30"><P class="p55 ft8">19</P></TD>
</TR>
<TR>
	<TD class="tr1 td31"><P class="p50 ft26">&nbsp;</P></TD>
	<TD class="tr1 td32"><P class="p50 ft26">&nbsp;</P></TD>
	<TD class="tr1 td33"><P class="p50 ft26">&nbsp;</P></TD>
</TR>
</TABLE>
<P class="p129 ft8">Table. 3. Results for <NOBR>Dataset–3</NOBR> (top), Results for <NOBR>Dataset–4</NOBR> (bottom).</P>
<P class="p130 ft6"><SPAN class="ft6">V.</SPAN><SPAN class="ft57">CONCLUSION</SPAN></P>
<P class="p131 ft9">According to our observation, choice of feature extractors and feature detectors is specific to the system. Previous studied showed SURF to be performing than SIFT. However we believe that it is true only in the case of Feature Detection. Thereby, from our observations, the fusion of SURF feature detection and SIFT feature extractor is the most efficient. However, it also has the higher time complexity. Therefore, with parallelization SURF <NOBR>–SURF</NOBR> performs optimally.</P>
<P class="p132 ft9">Additionally, for images with motion effects, GFFT perform worse than otherwise. To our knowledge, GFFT is the fastest performing feature detector. It produces optimal results for data without motion effects.</P>
<P class="p133 ft9">For our future work, we will extend Bag of Words with VLDA encoding and implement Spatial Pyramid Pooling. Also, we will try to uses CNN with Spatial Pyramid pooling technique to achieve better scene classification accuracy.</P>
<P class="p134 ft6">ACKNOWLEDGMENT</P>
</DIV>
<DIV id="id_1_2">
<P class="p135 ft6">VII. REFERENCES</P>
<P class="p136 ft59"><SPAN class="ft29">[1]</SPAN><SPAN class="ft58">A. Vailaya, M. Figueiredo, A. Jain, and H.J. Zhang. Image classification for </SPAN><NOBR>content-based</NOBR> indexing. IEEE Trans. on Image Processing, <NOBR>10(1):117–130,</NOBR> 2001.</P>
<P class="p137 ft59"><SPAN class="ft29">[2]</SPAN><SPAN class="ft58">Performance Comparison of Various Feature </SPAN><NOBR>Detector-Descriptor</NOBR> Combinations for <NOBR>Content-based</NOBR> Image Retrieval with <NOBR>JPEG-encoded</NOBR> Query Images by Jianshu Chao, Anas <NOBR>Al-Nuaimi,</NOBR> Georg Schroth and Eckehard Steinbach</P>
<P class="p138 ft59"><SPAN class="ft29">[3]</SPAN><SPAN class="ft58">S. Lazebnik, C. Schmid, J. Ponce. 2006. Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories. IEEE Computer Vision and Pattern Recognition, pp. </SPAN><NOBR>2169-2178.</NOBR></P>
<P class="p139 ft29"><SPAN class="ft60">[4]</SPAN><SPAN class="ft61">F. Monay, P. Quelhas, </SPAN><NOBR>J.-M.</NOBR> Odobez, and D. <NOBR>Gatica-Perez.</NOBR> Integrating <NOBR>co-occurrence</NOBR> and spatial contexts on patchbased scene segmentation. In CVPR, Beyond Patches Workshop, New York, NY, June <NOBR>17–22,</NOBR> 2006.</P>
<P class="p136 ft63"><SPAN class="ft29">[5]</SPAN><SPAN class="ft62">M. R. Boutell, J. Luo, and C. M. Brown. Factor graphs for </SPAN><NOBR>region-based</NOBR> <NOBR>whole-scene</NOBR> classification. In CVPR, Semantic Learning Workshop, New York, NY, June <NOBR>17–22,</NOBR> 2006.</P>
<P class="p136 ft59"><SPAN class="ft29">[6]</SPAN><SPAN class="ft58">J. C. van Gemert, J. Geusebroek, C. J. Veenman, C. G. M. Snoek, and A. W. M. Smeulders. Robust scene categorization by learning image statistics in context. In CVPR, Semantic Learning Workshop, New York, NY, June </SPAN><NOBR>17–22,</NOBR> 2006.</P>
<P class="p140 ft29"><SPAN class="ft60">[7]</SPAN><SPAN class="ft61">Performance Analysis of Various Feature Detector and Descriptor for </SPAN><NOBR>Real-Time</NOBR> Video based Face Tracking by Akash Patel, D. R. Kasat.</P>
<P class="p141 ft65"><SPAN class="ft29">[8]</SPAN><SPAN class="ft64">Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool </SPAN><A href="http://www.vision.ee.ethz.ch/~surf/papers.html">"SURF: Speeded Up Robust Features", </A>Computer Vision and Image Understanding (CVIU), Vol. 110, No. 3, pp. <NOBR>346–359,</NOBR> 2008.</P>
<P class="p136 ft59"><SPAN class="ft29">[9]</SPAN><SPAN class="ft66">E. Rosten and T. Drummond, “Machine learning for high speed corner detection,” in 9th Euproean Conference on Computer Vision, vol. 1, 2006, pp. </SPAN><NOBR>430–443.</NOBR></P>
<P class="p136 ft63"><SPAN class="ft29">[10]</SPAN><SPAN class="ft67">J. Matas, O. Chum, M. Urban, and T. Pajdla. Robust wide baseline stereo from maximally stable extremal regions. In Proc. of British Machine Vision Conference, pages </SPAN><NOBR>384–396,</NOBR> 2002.</P>
<P class="p142 ft59"><SPAN class="ft29">[11]</SPAN><SPAN class="ft68">Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary Bradski </SPAN><A href="http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf">"ORB: an efﬁcient alternative to SIFT or SURF", </A>Computer Vision (ICCV), 2011 IEEE International Conference on. IEEE, 2011.</P>
<P class="p143 ft63"><SPAN class="ft29">[12]</SPAN><SPAN class="ft67">Refer: </SPAN><A href="http://ttic.uchicago.edu/~mostajabi/Tutorial.html.%20By">http://ttic.uchicago.edu/~mostajabi/Tutorial.html. By </A>Mostajabi, 2011.</P>
<P class="p144 ft63"><SPAN class="ft29">[13]</SPAN><NOBR><SPAN class="ft67">L.-J.</SPAN></NOBR> Li and L. <NOBR>Fei-Fei.</NOBR> What, where and who?Classifying events by scene and object recognition. In ICCV, 2007.</P>
<P class="p145 ft63"><SPAN class="ft29">[14]</SPAN><SPAN class="ft67">A. Quattoni and A. Torralba. Recognizing indoor scenes. In CVPR, 2009.</SPAN></P>
<P class="p146 ft27"><SPAN class="ft29">[15]</SPAN><SPAN class="ft69">S. Lazebnik, C. Schmid, and J. Ponce. Beyond bags of</SPAN></P>
<P class="p147 ft27">features: Spatial pyramid matching for recognizing natural scene categories. In CVPR, 2006.</P>
</DIV>
</DIV>
<DIV id="id_2">
<P class="p148 ft9">Thanks to Dr. Dapeng Wu for supporting us and assisting in successful completion of the project. Thanks to researcher, contributing to the field of Scene classification and comparison of Feature extractors and Feature detectors.</P>
<P class="p149 ft12">8</P>
</DIV>
</DIV>
</BODY>
</HTML>
