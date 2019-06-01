#ifndef ORBSLAM2_ROS_FISHEYE_NODE_H
#define ORBSLAM2_ROS_FISHEYE_NODE_H

#include <string>
#include <fisheye_common/undistort.h>
#include <vector>

 #define ORB_EXPERIMENT_0601

class FisheyeNode {
 public:
  void RunFromDataset();
  std::string vocabulary_path;
  std::string settings_path;
  fisheye::FisheyeUndistort undistort;
 private:
#ifdef ORB_EXPERIMENT_0601
  void MatchKeypoints(
      // train
      const cv::Mat& train,
      cv::Mat& train_desc,
      // query
      const cv::Mat& query,
      std::vector<cv::KeyPoint>& all_kps,
      std::vector<cv::KeyPoint>& good_kps,
      cv::Mat& all_desc,
      cv::Mat& good_desc);
  void ORBExperiment0601();
  std::vector<cv::Mat> mono_frames_;
//  std::vector<std::vector<cv::KeyPoint>> mono_kps_;
//  std::vector<cv::Mat> mono_desc_;

  std::vector<std::pair<cv::Mat, cv::Mat>> stereo_frames_;
#endif
};

#endif // ORBSLAM2_ROS_FISHEYE_NODE_H
