#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

VideoWriter outputVideo;
Mat img_matrix;

void init_video_save(unsigned int size_x, unsigned int fps, char *video_filename)
{
	cv::Size vid_size;
	vid_size.width = size_x;
	vid_size.height = size_x;
	outputVideo.open(video_filename, CV_FOURCC('M','J','P','G') , fps, vid_size, true );

	if (!outputVideo.isOpened())
    {
        printf("Could not open the output video for write: ");
    }
}

void add_picture(unsigned char *img_buffer, unsigned int size_x)
{
	img_matrix = Mat(size_x , size_x, CV_8UC3, img_buffer);
	outputVideo.write(img_matrix);
}
