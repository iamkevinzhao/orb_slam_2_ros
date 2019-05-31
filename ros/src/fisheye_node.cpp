#include "fisheye_node.h"
#include "System.h"
#include <chrono>

void FisheyeNode::RunFromDataset() {
  ORB_SLAM2::System SLAM(
      vocabulary_path, settings_path, ORB_SLAM2::System::MONOCULAR);
  long now =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
  cv::Mat Tcw;
  int id = 0;
  while(true) {
    ++id;
    cv::Mat image;
    image =
        cv::imread(
            ("./images/mono_" + std::to_string(id) + ".jpg").c_str(),
            CV_LOAD_IMAGE_COLOR);

  }
}

int main() {

}
