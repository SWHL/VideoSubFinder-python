#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
  String str = "tmp/1.jpg";
  Mat image = imread(str);

  assert(image.rows == 80);
  std::cout << image.rows << std::endl;
  return 0;
}