#include "fisheye_node.h"
#include "System.h"
#include <chrono>

void FisheyeNode::RunFromDataset() {
  ORB_SLAM2::System slam(
      vocabulary_path, settings_path, ORB_SLAM2::System::MONOCULAR);
  long now =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
  int id = 0;
  while(true) {
    ++id;
    cv::Mat image;
    image =
        cv::imread(
            ("./images/mono_" + std::to_string(id) + ".jpg").c_str(),
            CV_LOAD_IMAGE_COLOR);
    image = undistort.Undistort(image);
    slam.TrackMonocular(image, now / 1000.0);
    cv::Mat debug_image = slam.DrawCurrentFrame();
    cv::imshow("debug", debug_image);
    if (cv::waitKey(1) >= 0) {
      break;
    }
  }
  slam.Shutdown();
}

int main() {
  FisheyeNode node;
  node.vocabulary_path = "ORBvoc.txt";
  node.settings_path = "webcam.yaml";
  if (!node.undistort.LoadCalibResult("calib_result.txt")) {
    return -1;
  }
  node.RunFromDataset();
}
