#include "fisheye_node.h"
#include "System.h"
#include <chrono>
#include <fisheye_common/fisheye_common.h>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#ifdef QT5CHARTS_FOUND
#include <QtCharts/QtCharts>
#endif

void PutText(cv::Mat& image, const std::vector<std::string>& texts) {
  int vpos = 0;
  for (const std::string& text : texts) {
    vpos += 17;
    cv::putText(image, text, {4, vpos}, 1, 1.3, {0, 0, 0});
  }
}

void FisheyeNode::RunInRealTime() {
  ORB_SLAM2::System slam(
      vocabulary_path, settings_path, ORB_SLAM2::System::MONOCULAR);
  cv::VideoCapture cap(1);
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
//  cap.set(CV_CAP_PROP_EXPOSURE, -7);

//  stereo.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
//  stereo.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  int id = 0;
  while(true) {
    ++id;
    cv::Mat mono;
    cap >> mono;
    if (mono.empty()) {
      continue;
    }
//    cv::imwrite(("./images/mono_" + std::to_string(id) + ".jpg").c_str(), mono);
    mono = undistort.Undistort(mono);
    // cv::imshow("test", mono);
//    cv::Mat stereo;
//    stereo =
//        cv::imread(
//            ("./images/stereo_" + std::to_string(id) + ".jpg").c_str(),
//            CV_LOAD_IMAGE_COLOR);
//    std::pair<cv::Mat, cv::Mat> stereo_pair =
//        undistort.Undistort(fisheye::Split(stereo));
//    cv::imshow("left", stereo_pair.first);
//    cv::imshow("right", stereo_pair.second);

    long now =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    slam.TrackMonocular(mono, now / 1000.0);
    cv::Mat debug_image = slam.DrawCurrentFrame();
    cv::imshow("debug", debug_image);
    if (cv::waitKey(1) >= 0) {
      break;
    }
  }
  slam.Shutdown();
}

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
    cv::imwrite(("mono_" + std::to_string(id)).c_str(), mono);
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
  // node.RunInRealTime();
  // node.RunFromDataset();
  node.RunORBExperiment0601();
}

///////////////////////////////////////////////////////////////////

#ifdef ORB_EXPERIMENT_0601
void FisheyeNode::RunORBExperiment0601() {
  int id = 0;
  while(true) {
    ++id;
    cv::Mat mono;
    mono =
        cv::imread(
            ("./images/mono_" + std::to_string(id) + ".jpg").c_str(),
            CV_LOAD_IMAGE_COLOR);
    mono = undistort.Undistort(mono);

    cv::Mat stereo;
    stereo =
        cv::imread(
            ("./images/stereo_" + std::to_string(id) + ".jpg").c_str(),
            CV_LOAD_IMAGE_COLOR);
    std::pair<cv::Mat, cv::Mat> stereo_pair =
        undistort.Undistort(fisheye::Split(stereo));
//    cv::imshow("left", stereo_pair.first);
//    cv::imshow("right", stereo_pair.second);

    if (!matcher_) {
      matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");
    }
    if (!detector_) {
      detector_ = cv::ORB::create();
    }
    mono_frames_.push_back(mono.clone());
    stereo_frames_.push_back(stereo_pair);
    // ORBExperiment0601();
    ORBExperiment0602();
    if (mono_frames_.size() >= 3) {
      mono_frames_.erase(mono_frames_.begin());
    }
    stereo_frames_.erase(stereo_frames_.begin());

    if (cv::waitKey(1) >= 0) {
      break;
    }
  }
}

void FisheyeNode::ORBExperiment0601() {
  if (mono_frames_.size() < 3) {
    return;
  }

  using namespace std;
  using namespace cv;

  vector<KeyPoint> kp1, kp2, kp3;
  cv::Mat dp1, dp2, dp3;
  cv::Mat& f1 = mono_frames_[0], f2 = mono_frames_[1], f3 = mono_frames_[2];
  detector_->detectAndCompute(f1, noArray(), kp1, dp1);
  detector_->detectAndCompute(f2, noArray(), kp2, dp2);
  detector_->detectAndCompute(f3, noArray(), kp3, dp3);
  vector<DMatch> matches;
  vector<KeyPoint> kp3in2;
  matcher_->match(dp3, dp2, matches);
  for (DMatch& match : matches) {
    if (match.distance < kThres) {
      kp3in2.push_back(kp3[match.queryIdx]);
    }
  }
  Mat dp1in2;
  matcher_->match(dp1, dp2, matches);
  for (DMatch& match : matches) {
    if (match.distance < kThres) {
      if (dp1in2.empty()) {
        dp1in2 = dp1.row(match.queryIdx).clone();
      } else {
        vconcat(dp1in2, dp1.row(match.queryIdx), dp1in2);
      }
    }
  }
  vector<KeyPoint> kp3in1;
  matcher_->match(dp3, dp1in2, matches);
  for (DMatch& match : matches) {
    if (match.distance < kThres) {
      kp3in1.push_back(kp3[match.queryIdx]);
    }
  }
  Mat image;
  drawKeypoints(f3, kp3, image, {0, 255, 0});
  drawKeypoints(image, kp3in2, image, {255, 0, 0});
  drawKeypoints(image, kp3in1, image, {0, 0, 255});
  PutText(
      image,
      {"ORB in green: " + to_string(kp3.size()),
       "ORB in blue: " + to_string(kp3in2.size()),
       "ORB in red: " + to_string(kp3in1.size())});
  imshow("keypoints", image);

  static std::vector<std::tuple<int, int, int>> stream(
      kSteamBufSize, std::make_tuple(0, 0, 0));
  stream.push_back(std::make_tuple(kp3.size(), kp3in2.size(), kp3in1.size()));
  stream.erase(stream.begin());

  static ofstream os;
  if (!os.is_open()) {
    os.open("statistics_mono.txt");
  }
  os << kp3.size() << " " << kp3in2.size() << " " << kp3in1.size() << "\n";
#ifdef QT5CHARTS_FOUND
  static QChart *chart = new QChart();
  static QChartView *chartView = new QChartView(chart);
  chartView->show();
  static QLineSeries *green_series = new QLineSeries();
  static QLineSeries *blue_series = new QLineSeries();
  static QLineSeries *red_series = new QLineSeries();
  static QValueAxis *axis_y = new QValueAxis;
  axis_y->setMin(0);
  axis_y->setMax(600);
  axis_y->setLabelFormat("%d");
  // axis_y->setTickCount(20);
//  chartView->setSize(500, 500);
  chartView->setWindowTitle("ORB Feature Statistics (Mono)");
  if (chart->axes().empty()) {
    chart->addAxis(axis_y, Qt::AlignRight);
  }
  if (green_series->attachedAxes().empty()) {
    green_series->attachAxis(axis_y);
    blue_series->attachAxis(axis_y);
    red_series->attachAxis(axis_y);
  }
  green_series->setColor(Qt::green);
  blue_series->setColor(Qt::blue);
  red_series->setColor(Qt::red);

  green_series->clear();
  blue_series->clear();
  red_series->clear();
  int i = 0;
  for (auto& tuple : stream) {
    ++i;
    green_series->append(i, get<0>(tuple));
    blue_series->append(i, get<1>(tuple));
    red_series->append(i, get<2>(tuple));
  }
  if (chart->series().empty()) {
    chart->addSeries(green_series);
    chart->addSeries(blue_series);
    chart->addSeries(red_series);
  }
#endif
}

void FisheyeNode::ORBExperiment0602() {
  if (mono_frames_.empty()) {
    return;
  }
  if (stereo_frames_.empty()) {
    return;
  }
  using namespace std;
  using namespace cv;
  constexpr float ratio = 0.21;
  constexpr int kThres = 40;

  Mat& mono = *mono_frames_.rbegin();
  Mat& stereo_left = stereo_frames_.rbegin()->first;
  Mat& stereo_right = stereo_frames_.rbegin()->second;

  Mat mo_roi =
      mono(Rect(0, mono.rows * (1 - ratio), mono.cols, mono.rows * ratio));
  Mat sol_roi =
      stereo_left(Rect(0, 0, stereo_left.cols, stereo_left.rows * ratio));
  Mat sor_roi =
        stereo_right(Rect(0, 0, stereo_right.cols, stereo_right.rows * ratio));
  vector<KeyPoint> mo_kp, sol_kp, sor_kp;
  Mat mo_dp, sol_dp, sor_dp;
  detector_->detectAndCompute(mo_roi, noArray(), mo_kp, mo_dp);
  detector_->detectAndCompute(sol_roi, noArray(), sol_kp, sol_dp);
  detector_->detectAndCompute(sor_roi, noArray(), sor_kp, sor_dp);
  vector<KeyPoint> sol_cokp, sor_cokp;
  vector<DMatch> matches;
  cv::Mat sol_codp, sor_codp;
  matcher_->match(sol_dp, mo_dp, matches);
  for (DMatch& match : matches) {
    if (match.distance > kThres) {
      continue;
    }
    sol_cokp.push_back(sol_kp[match.queryIdx]);
    if (sol_codp.empty()) {
      sol_codp = sol_dp.row(match.queryIdx).clone();
    } else {
      vconcat(sol_codp, sol_dp.row(match.queryIdx), sol_codp);
    }
  }
  matcher_->match(sor_dp, mo_dp, matches);
  for (DMatch& match : matches) {
    if (match.distance > kThres) {
      continue;
    }
    sor_cokp.push_back(sor_kp[match.queryIdx]);
    if (sor_codp.empty()) {
      sor_codp = sor_dp.row(match.queryIdx).clone();
    } else {
      vconcat(sor_codp, sor_dp.row(match.queryIdx), sor_codp);
    }
  }
  std::vector<KeyPoint> sol_3kp, sor_3kp;
  if (!sol_codp.empty() && !sor_codp.empty()) {
    matcher_->match(sol_codp, sor_codp, matches);
    for (DMatch& match : matches) {
      if (match.distance > 70) {
        continue;
      }
      sol_3kp.push_back(sol_cokp[match.queryIdx]);
      sor_3kp.push_back(sor_cokp[match.trainIdx]);
    }
  }
  drawKeypoints(mo_roi, mo_kp, mo_roi, Scalar{0, 255, 0});
  drawKeypoints(sol_roi, sol_kp, sol_roi, Scalar{0, 255, 0});
  drawKeypoints(sor_roi, sor_kp, sor_roi, Scalar{0, 255, 0});
  drawKeypoints(sol_roi, sol_cokp, sol_roi, Scalar{255, 0, 0});
  drawKeypoints(sor_roi, sor_cokp, sor_roi, Scalar{255, 0, 0});
  drawKeypoints(sol_roi, sol_3kp, sol_roi, Scalar{0, 0, 255});
  drawKeypoints(sor_roi, sor_3kp, sor_roi, Scalar{0, 0, 255});
  PutText(mo_roi, {"ORB in green: " + to_string(mo_kp.size())});
  PutText(
      sol_roi,
      {"ORB in green: " + to_string(sol_kp.size()),
       "ORB in blue: " + to_string(sol_cokp.size()),
       "ORB in red: " + to_string(sol_3kp.size())});
  PutText(
      sor_roi,
      {"ORB in green: " + to_string(sor_kp.size()),
       "ORB in blue: " + to_string(sor_cokp.size()),
       "ORB in red: " + to_string(sor_3kp.size())});
  cv::Mat image = mo_roi.clone();
  vconcat(image, sol_roi, image);
  vconcat(image, sor_roi, image);

  imshow("mono-stereo", image);

  ofstream os;
  if (!os.is_open()) {
    os.open("statistics_stereo.txt");
  }
  os << mo_kp.size() << " "
     << sol_3kp.size() << " "
     << sol_cokp.size() << " "
     << sol_3kp.size() << " "
     << sor_3kp.size() << " "
     << sor_cokp.size() << " "
     << sor_3kp.size() << "\n";

  static std::vector<std::tuple<int>> stream(
      kSteamBufSize, std::make_tuple(0));
  stream.push_back(std::make_tuple(sor_3kp.size()));
  stream.erase(stream.begin());

#ifdef QT5CHARTS_FOUND
  static QChart *chart = new QChart();
  static QChartView *chartView = new QChartView(chart);
  chartView->show();
  static QLineSeries *red_series = new QLineSeries();
  static QValueAxis *axis_y = new QValueAxis;
  axis_y->setMin(0);
  axis_y->setMax(30);
  axis_y->setLabelFormat("%d");
//  axis_y->setTickCount(28);
//  chartView->setSize(500, 500);
  chartView->setWindowTitle("ORB Feature Statistics (Stereo)");
  if (chart->axes().empty()) {
    chart->addAxis(axis_y, Qt::AlignRight);
  }
  if (red_series->attachedAxes().empty()) {
    red_series->attachAxis(axis_y);
  }
  red_series->setColor(Qt::red);

  red_series->clear();
  int i = 0;
  for (auto& tuple : stream) {
    ++i;
    red_series->append(i, get<0>(tuple));
  }
  if (chart->series().empty()) {
    chart->addSeries(red_series);
  }
#endif
}
#endif
