////////////////////////////////////////////////////////////////
/////////////////////code by Arpita S Tugave
////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp> 
#include <time.h>

using namespace cv;

void readme();

/** @function main */
int main( int argc, char** argv )
{
    //set timer
    clock_t t1,t2;
    t1=clock();
  
  //read input arguments
  if( argc != 2 )
  { readme(); return -1; }

  //read image
  Mat img_1 = imread( "lena.bmp", CV_LOAD_IMAGE_GRAYSCALE );

  //error reading message
  if( !img_1.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	//initialize keypoints
	std::vector<KeyPoint> keypoints_1; 
	Mat img_keypoints_1; 

	if(string(argv[1]).compare("SURF") == 0)
	{
	    cv::DescriptorExtractor * extractor = new cv::SURF();
	    cv::FeatureDetector * detector = new cv::SURF(500.0);

	//detect and extract
	detector->detect( img_1, keypoints_1 );
  	extractor->compute(img_1,keypoints_1, img_keypoints_1);

	//draw extracted key points on image
        drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//show the time taken to extract
	t2=clock();
	float diff ((float)t2-(float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout<<seconds<<std::endl;

        //-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1 );
	imwrite("SURF.png", img_keypoints_1 );
	waitKey(0);

	}
	else if (string(argv[1]).compare("SIFT") == 0)
	{
            cv::DescriptorExtractor * extractor = new cv::SIFT();
	    cv::FeatureDetector * detector = new cv::SIFT();

	//detect and extract
	detector->detect( img_1, keypoints_1 );
  	extractor->compute(img_1,keypoints_1, img_keypoints_1);

	//draw extracted key points on image
        drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//show the time taken to extract
	t2=clock();
	float diff ((float)t2-(float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout<<seconds<<std::endl;

        //-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1 );
	imwrite("SIFT.png", img_keypoints_1 );
	waitKey(0);
	}
	else if (string(argv[1]).compare("FAST") == 0)
	{
	    cv::DescriptorExtractor * extractor = new cv::SIFT();
	    cv::FeatureDetector * detector = new cv::FastFeatureDetector();
	//detect and extract
	detector->detect( img_1, keypoints_1 );
  	extractor->compute(img_1,keypoints_1, img_keypoints_1);

	//draw extracted key points on image
        drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//show the time taken to extract
	t2=clock();
	float diff ((float)t2-(float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout<<seconds<<std::endl;

        //-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1 );
	imwrite("FAST.png", img_keypoints_1 );
	waitKey(0);

	}
	else if (string(argv[1]).compare("BRISK") == 0)
	{
	    cv::DescriptorExtractor * extractor = new cv::SIFT();
	    cv::FeatureDetector * detector = new cv::BRISK();

	//detect and extract
	detector->detect( img_1, keypoints_1 );
  	extractor->compute(img_1,keypoints_1, img_keypoints_1);

	//draw extracted key points on image
        drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//show the time taken to extract
	t2=clock();
	float diff ((float)t2-(float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout<<seconds<<std::endl;

        //-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1 );
	imwrite("BRISK.png", img_keypoints_1 );
	waitKey(0);
	}
	else if (string(argv[1]).compare("STAR") == 0)
	{
	    cv::DescriptorExtractor * extractor = new cv::SIFT();
	    cv::FeatureDetector * detector = new cv::StarFeatureDetector();

	//detect and extract
	detector->detect( img_1, keypoints_1 );
  	extractor->compute(img_1,keypoints_1, img_keypoints_1);

	//draw extracted key points on image
        drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//show the time taken to extract
	t2=clock();
	float diff ((float)t2-(float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout<<seconds<<std::endl;

        //-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1 );
	imwrite("STAR.png", img_keypoints_1 );
	waitKey(0);

	}
	else if (string(argv[1]).compare("MSER") == 0)
	{
	    cv::DescriptorExtractor * extractor = new cv::SIFT();
	    cv::FeatureDetector * detector = new cv::MSER();

	//detect and extract
	detector->detect( img_1, keypoints_1 );
  	extractor->compute(img_1,keypoints_1, img_keypoints_1);

	//draw extracted key points on image
        drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//show the time taken to extract
	t2=clock();
	float diff ((float)t2-(float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout<<seconds<<std::endl;

        //-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1 );
	imwrite("MSER.png", img_keypoints_1 );
	waitKey(0);

	}
	else if (string(argv[1]).compare("GFFT") == 0)
	{
	    cv::DescriptorExtractor * extractor = new cv::SIFT();
	    cv::FeatureDetector * detector = new cv::GFTTDetector();

	//detect and extract
	detector->detect( img_1, keypoints_1 );
  	extractor->compute(img_1,keypoints_1, img_keypoints_1);

	//draw extracted key points on image
        drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//show the time taken to extract
	t2=clock();
	float diff ((float)t2-(float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout<<seconds<<std::endl;

        //-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1 );
	imwrite("GFFT.png", img_keypoints_1 );
	waitKey(0);


	}
	else if (string(argv[1]).compare("DENSE") == 0)
	{
	    cv::DescriptorExtractor * extractor = new cv::SIFT();
	    cv::FeatureDetector * detector = new cv::DenseFeatureDetector();

	//detect and extract
	detector->detect( img_1, keypoints_1 );
  	extractor->compute(img_1,keypoints_1, img_keypoints_1);

	//draw extracted key points on image
        drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//show the time taken to extract
	t2=clock();
	float diff ((float)t2-(float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout<<seconds<<std::endl;

        //-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1 );
	imwrite("DENSE.png", img_keypoints_1 );
	waitKey(0);

	}
	else
	{
	std::cout << "feature not detected" << std::endl;
	}
	
  return 0;
  }

  /** @function readme */
  void readme()
  { std::cout << " Usage: ./feature <feature> " << std::endl; }

