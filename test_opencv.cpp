#include <opencv2/opencv.hpp>

using namespace cv;
void ImageThreshold(String str) {
  Mat image = imread(str);

  imshow("test_opencv",image);
  waitKey(0);
}
int main() {
  String str = "/Users/jiahuawang/projects/VideoSubFinder-python/tmp/1.jpg";
  ImageThreshold(str);
  return 0;
}