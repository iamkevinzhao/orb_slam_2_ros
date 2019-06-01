#include "fisheye_node.h"
#include "System.h"
#include <chrono>
#include <fisheye_common/fisheye_common.h>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

void FisheyeNode::RunFromDataset() {
  ORB_SLAM2::System slam(
      vocabulary_path, settings_path, ORB_SLAM2::System::MONOCULAR);
  long now =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
  int id = 0;
  while(true) {
    ++id;
    cv::Mat mono;
    mono =
        cv::imread(
            ("./images/mono_" + std::to_string(id) + ".jpg").c_str(),
            CV_LOAD_IMAGE_COLOR);
    mono = undistort.Undistort(mono);

//    cv::Mat stereo;
//    stereo =
//        cv::imread(
//            ("./images/stereo_" + std::to_string(id) + ".jpg").c_str(),
//            CV_LOAD_IMAGE_COLOR);
//    std::pair<cv::Mat, cv::Mat> stereo_pair =
//        undistort.Undistort(fisheye::Split(stereo));
//    cv::imshow("left", stereo_pair.first);
//    cv::imshow("right", stereo_pair.second);

#ifdef ORB_EXPERIMENT_0601
    mono_frames_.push_back(mono.clone());
    // stereo_frames_.push_back(stereo_pair);
    ORBExperiment0601();
    mono_frames_.erase(mono_frames_.begin());
#endif

    slam.TrackMonocular(mono, now / 1000.0);
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


#ifdef ORB_EXPERIMENT_0601
void FisheyeNode::ORBExperiment0601() {
  if (mono_frames_.size() < 3) {
    return;
  }
  static cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  static cv::Ptr<cv::ORB> detector = cv::ORB::create();

  constexpr int thres = 30;

  using namespace std;
  using namespace cv;
  vector<KeyPoint> kp1, kp2, kp3;
  cv::Mat dp1, dp2, dp3;
  cv::Mat& f1 = mono_frames_[0], f2 = mono_frames_[1], f3 = mono_frames_[2];
  detector->detectAndCompute(f1, noArray(), kp1, dp1);
  detector->detectAndCompute(f2, noArray(), kp2, dp2);
  detector->detectAndCompute(f3, noArray(), kp3, dp3);
  vector<DMatch> matches;
  vector<KeyPoint> kp3in2;
  matcher->match(dp3, dp2, matches);
  for (DMatch& match : matches) {
    if (match.distance < thres) {
      kp3in2.push_back(kp3[match.queryIdx]);
    }
  }
  cv::Mat dp1in2;
  matcher->match(dp1, dp2, matches);
  for (DMatch& match : matches) {
    if (match.distance < thres) {
      if (dp1in2.empty()) {
        dp1in2 = dp1.row(match.queryIdx).clone();
      } else {
        vconcat(dp1in2, dp1.row(match.queryIdx), dp1in2);
      }
    }
  }
  vector<KeyPoint> kp3in1;
  matcher->match(dp3, dp1in2, matches);
  for (DMatch& match : matches) {
    if (match.distance < thres) {
      kp3in1.push_back(kp3[match.queryIdx]);
    }
  }
  cv::Mat image;
  drawKeypoints(f3, kp3, image, {0, 255, 0});
  drawKeypoints(image, kp3in2, image, {255, 0, 0});
  drawKeypoints(image, kp3in1, image, {0, 0, 255});
  imshow("keypoints", image);
}
#endif

//////////////////////////////////////////////////////////////
#if 0
void FisheyeNode::MatchKeypoints(
    // train
    const cv::Mat& train,
    cv::Mat& train_desc,
    // query
    const cv::Mat& query,
    std::vector<cv::KeyPoint>& all_kps,
    std::vector<cv::KeyPoint>& good_kps,
    cv::Mat& all_desc,
    cv::Mat& good_desc) {
  static cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  static cv::Ptr<cv::ORB> detector = cv::ORB::create();

  std::vector<cv::KeyPoint> kp_tmp;
  if (train_desc.empty()) {
    detector->detectAndCompute(
        train, cv::noArray(), kp_tmp, train_desc);
//    good_kps = all_kps;
//    good_desc = train_desc;
//    return;
  }

  if (all_desc.empty()) {
    detector->detectAndCompute(query, cv::noArray(), all_kps, all_desc);
  }

  std::vector<cv::DMatch> matches, good_matches;
  // std::cout << all_desc.size << " " << train_desc.size << " " << matches.size() << std::endl;
  matcher->match(all_desc, train_desc, matches);
  for (cv::DMatch& match : matches) {
    if (match.distance <= 30) {
      good_matches.push_back(match);
    }
  }
  for (cv::DMatch& match : good_matches) {
    int idx = match.queryIdx;
    good_kps.push_back(all_kps[idx]);
    // std::cout << good_desc.size << " " << all_desc.row(idx).size << std::endl;
    if (good_desc.empty()) {
      good_desc = all_desc.row(idx).clone();
    }
    cv::vconcat(good_desc, all_desc.row(idx), good_desc);
  }
  return;
}

void FisheyeNode::ORBExperiment0601() {
  if (mono_frames_.size() < 3) {
    mono_desc_.assign(3, cv::Mat());
    mono_kps_.assign(3, std::vector<cv::KeyPoint>());
    return;
  }
  mono_desc_.push_back(cv::Mat());
  mono_kps_.push_back(std::vector<cv::KeyPoint>());
  if (mono_frames_.size() > 3) {
    mono_frames_.erase(mono_frames_.begin());
    mono_desc_.erase(mono_desc_.begin());
    mono_kps_.erase(mono_kps_.begin());
  }
//  cv::Mat& frame_1, frame_2, frame_3;
//  cv::Mat& desc_1, desc_2, desc_3;
//  std::vector<cv::KeyPoint>& kp_1, kp_2, kp_3;
  cv::Mat& frame_1 = mono_frames_[0];
  cv::Mat& frame_2 = mono_frames_[1];
  cv::Mat& frame_3 = mono_frames_[2];
  cv::Mat& desc_1 = mono_desc_[0];
  cv::Mat& desc_2 = mono_desc_[1];
  cv::Mat& desc_3 = mono_desc_[2];
  std::vector<cv::KeyPoint>& kp_1 = mono_kps_[0];
  std::vector<cv::KeyPoint>& kp_2 = mono_kps_[1];
  std::vector<cv::KeyPoint>& kp_3 = mono_kps_[2];

  std::vector<cv::KeyPoint> kp_all, kp_31;
  cv::Mat all_desc, desc_31;
  MatchKeypoints(frame_1, desc_1, frame_3, kp_all, kp_31, all_desc, desc_31);
  MatchKeypoints(frame_2, desc_2, frame_3, kp_all, kp_3, all_desc, desc_3);
  cv::Mat image = frame_3.clone();
  cv::drawKeypoints(image, kp_all, image, {255, 0, 0});
  cv::drawKeypoints(image, kp_3, image, {0, 0, 255});
  // cv::drawKeypoints(image, kp_31, image, {0, 255, 0});
  cv::imshow("keypoints", image);
}

//void FisheyeNode::ORBExperiment0601() {
//  if (mono_frames_.size() < 3) {
//    return;
//  }
//  static cv::Ptr<cv::DescriptorMatcher> matcher =
//      cv::DescriptorMatcher::create("BruteForce-Hamming");
//  static cv::Ptr<cv::ORB> detector = cv::ORB::create();
//  cv::Mat& mono_1 = *mono_frames_.begin();
//  cv::Mat& mono_2 = *std::next(mono_frames_.begin());
//  cv::Mat& mono_3 = *std::next(mono_frames_.begin(), 2);

//  std::vector<cv::KeyPoint> kps_1, kps_2, kps_3;
//  cv::Mat desc_1, desc_2, desc_3;
//  if (mono_kps_.empty()) {
//    detector->detectAndCompute(mono_1, cv::noArray(), kps_1, desc_1);
//    detector->detectAndCompute(mono_2, cv::noArray(), kps_2, desc_2);
//    mono_kps_.push_back(kps_1);
//    mono_kps_.push_back(kps_2);
//    mono_desc_.push_back(desc_1);
//    mono_desc_.push_back(desc_2);
//  } else {
//    kps_1 = mono_kps_[0];
//    kps_2 = mono_kps_[1];
//    desc_1 = mono_desc_[0];
//    desc_2 = mono_desc_[1];
//  }
//  detector->detectAndCompute(mono_3, cv::noArray(), kps_3, desc_3);
//  mono_kps_.push_back(kps_3);
//  mono_desc_.push_back(desc_3);

//  std::vector<cv::DMatch>
//      matches_31, matches_32, good_matches_31, good_matches_32;
//  matcher->match(desc_3, desc_1, matches_31);
//  matcher->match(desc_3, desc_2, matches_32);
//  constexpr int thres = 30;
//  for (cv::DMatch& match : matches_31) {
//    if (match.distance <= thres) {
//      good_matches_31.push_back(match);
//    }
//  }
//  for (cv::DMatch& match : matches_32) {
//    if (match.distance <= thres) {
//      good_matches_32.push_back(match);
//    }
//  }
//}
#endif

///////////////////////////////////////////////////////////////
#if 0
void FisheyeNode::ORBExperiment0601() {
  if (mono_frames_.size() < 2 /*|| stereo_frames_.size() < 3*/) {
    return;
  }
//  std::pair<cv::Mat, cv::Mat> stereo = std::move(*stereo_frames_.rbegin());

  cv::Ptr<cv::ORB> detector = cv::ORB::create();

  cv::Mat mono_1 = std::move(*mono_frames_.begin());
  std::vector<cv::KeyPoint> keypoints_1;
  cv::Mat descriptors_1;
  detector->detectAndCompute(mono_1, cv::noArray(), keypoints_1, descriptors_1);

  cv::Mat mono_2 = std::prev(mono_frames_.end())->clone();
  std::vector<cv::KeyPoint> keypoints_2;
  cv::Mat descriptors_2;
  detector->detectAndCompute(mono_2, cv::noArray(), keypoints_2, descriptors_2);

  std::vector<cv::DMatch> matches;
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  matcher->match(descriptors_1, descriptors_2, matches);

//  cv::FlannBasedMatcher matcher;
//  std::vector<cv::DMatch> matches;
//  matcher.match(descriptors_1, descriptors_2, matches);

  double max_dist = 0, min_dist = 100;
  for (int i = 0; i < descriptors_1.rows; ++i) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  std::vector<cv::DMatch> good_matches;

  for (int i = 0; i < descriptors_1.rows; ++i) {
    // std::cout << matches[i].distance << " " << matches[i].imgIdx << std::endl;
    if (matches[i].distance <= 30) {
      good_matches.push_back(matches[i]);
    }
  }

  std::vector<cv::KeyPoint> good_keypoints;
  for (int i = 0; i < good_matches.size(); ++i) {
    good_keypoints.push_back(keypoints_1[good_matches[i].queryIdx]);
  }

  cv::drawKeypoints(mono_1, keypoints_1, mono_1, cv::Scalar{255, 0, 0});
  cv::drawKeypoints(mono_1, good_keypoints, mono_1, cv::Scalar{0, 0, 255});

  cv::imshow("keypoints", mono_1);

  // std::cout << keypoints_1.size() << " " << keypoints_2.size() << " " <<  good_matches.size() << std::endl;
  // std::cout << matches[0].queryIdx << " " << matches[0].trainIdx << std::endl;
  // std::cout << keypoints_1[matches[0].queryIdx].pt << " " << keypoints_2[matches[0].trainIdx].pt << std::endl;

  // std::cout << descriptors_1.size << std::endl;

  mono_frames_.erase(mono_frames_.begin());
//  mono_frames_.clear();
  stereo_frames_.clear();
//  mono_frames_.erase(mono_frames_.begin());
//  stereo_frames_.erase(stereo_frames_.begin());
}
#endif
