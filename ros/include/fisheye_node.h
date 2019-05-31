#ifndef ORBSLAM2_ROS_FISHEYE_NODE_H
#define ORBSLAM2_ROS_FISHEYE_NODE_H

#include <string>
#include <fisheye_common/undistort.h>

class FisheyeNode {
 public:
  void RunFromDataset();
  std::string vocabulary_path;
  std::string settings_path;
  fisheye::FisheyeUndistort undistort;
 private:
};

#endif // ORBSLAM2_ROS_FISHEYE_NODE_H
