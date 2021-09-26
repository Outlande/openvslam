#include "openvslam/camera/base.h"
#include "openvslam/data/common.h"
#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/camera_database.h"
#include "openvslam/data/map_database.h"
#include "openvslam/util/converter.h"
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace openvslam {
namespace data {

std::mutex map_database::mtx_database_;

map_database::map_database() {
    spdlog::debug("CONSTRUCT: data::map_database");
}

map_database::~map_database() {
    clear();
    spdlog::debug("DESTRUCT: data::map_database");
}

void map_database::add_keyframe(keyframe* keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    keyframes_[keyfrm->id_] = keyfrm;
    if (keyfrm->id_ > max_keyfrm_id_) {
        max_keyfrm_id_ = keyfrm->id_;
    }
}

void map_database::erase_keyframe(keyframe* keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    keyframes_.erase(keyfrm->id_);

    // TODO: delete object
}

void map_database::add_landmark(landmark* lm) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    landmarks_[lm->id_] = lm;
}

void map_database::erase_landmark(landmark* lm) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    landmarks_.erase(lm->id_);
    delete_from_grid(lm->id_, lm->last_grid_x_, lm->last_grid_y_);

    // TODO: delete object
}

void map_database::set_local_landmarks(const std::vector<landmark*>& local_lms) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    local_landmarks_ = local_lms;
}

std::vector<landmark*> map_database::get_local_landmarks() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return local_landmarks_;
}

std::vector<keyframe*> map_database::get_all_keyframes() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    std::vector<keyframe*> keyframes;
    keyframes.reserve(keyframes_.size());
    for (const auto id_keyframe : keyframes_) {
        keyframes.push_back(id_keyframe.second);
    }
    return keyframes;
}

std::vector<keyframe*> map_database::get_close_keyframes(const Mat44_t& pose,
                                                         const double distance_threshold,
                                                         const double angle_threshold) const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    // Close (within given thresholds) keyframes
    std::vector<keyframe*> filtered_keyframes;

    const double cos_angle_threshold = std::cos(angle_threshold);

    // Calculate angles and distances between given pose and all keyframes
    Mat33_t M = pose.block<3, 3>(0, 0);
    Vec3_t Mt = pose.block<3, 1>(0, 3);
    for (const auto id_keyframe : keyframes_) {
        Mat33_t N = id_keyframe.second->get_cam_pose().block<3, 3>(0, 0);
        Vec3_t Nt = id_keyframe.second->get_cam_pose().block<3, 1>(0, 3);
        // Angle between two cameras related to given pose and selected keyframe
        const double cos_angle = ((M * N.transpose()).trace() - 1) / 2;
        // Distance between given pose and selected keyframe
        const double dist = (Nt - Mt).norm();
        if (dist < distance_threshold && cos_angle > cos_angle_threshold) {
            filtered_keyframes.push_back(id_keyframe.second);
        }
    }

    return filtered_keyframes;
}

unsigned int map_database::get_num_keyframes() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return keyframes_.size();
}

std::vector<landmark*> map_database::get_all_landmarks() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    std::vector<landmark*> landmarks;
    landmarks.reserve(landmarks_.size());
    for (const auto id_landmark : landmarks_) {
        landmarks.push_back(id_landmark.second);
    }
    return landmarks;
}

unsigned int map_database::get_num_landmarks() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return landmarks_.size();
}

unsigned int map_database::get_max_keyframe_id() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return max_keyfrm_id_;
}

void map_database::clear() {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    for (auto& lm : landmarks_) {
        delete lm.second;
        lm.second = nullptr;
    }

    for (auto& keyfrm : keyframes_) {
        delete keyfrm.second;
        keyfrm.second = nullptr;
    }

    landmarks_.clear();
    keyframes_.clear();
    max_keyfrm_id_ = 0;
    local_landmarks_.clear();
    origin_keyfrm_ = nullptr;

    frm_stats_.clear();

    spdlog::info("clear map database");
}

void map_database::from_json(camera_database* cam_db, bow_vocabulary* bow_vocab, bow_database* bow_db,
                             const nlohmann::json& json_keyfrms, const nlohmann::json& json_landmarks) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    // Step 1. delete all the data in map database
    for (auto& lm : landmarks_) {
        delete lm.second;
        lm.second = nullptr;
    }

    for (auto& keyfrm : keyframes_) {
        delete keyfrm.second;
        keyfrm.second = nullptr;
    }

    landmarks_.clear();
    keyframes_.clear();
    max_keyfrm_id_ = 0;
    local_landmarks_.clear();
    origin_keyfrm_ = nullptr;

    // Step 2. Register keyframes
    // If the object does not exist at this step, the corresponding pointer is set as nullptr.
    spdlog::info("decoding {} keyframes to load", json_keyfrms.size());
    for (const auto& json_id_keyfrm : json_keyfrms.items()) {
        const auto id = std::stoi(json_id_keyfrm.key());
        assert(0 <= id);
        const auto json_keyfrm = json_id_keyfrm.value();

        register_keyframe(cam_db, bow_vocab, bow_db, id, json_keyfrm);
    }

    // Step 3. Register 3D landmark point
    // If the object does not exist at this step, the corresponding pointer is set as nullptr.
    spdlog::info("decoding {} landmarks to load", json_landmarks.size());
    for (const auto& json_id_landmark : json_landmarks.items()) {
        const auto id = std::stoi(json_id_landmark.key());
        assert(0 <= id);
        const auto json_landmark = json_id_landmark.value();

        register_landmark(id, json_landmark);
    }

    // Step 4. Register graph information
    spdlog::info("registering essential graph");
    for (const auto& json_id_keyfrm : json_keyfrms.items()) {
        const auto id = std::stoi(json_id_keyfrm.key());
        assert(0 <= id);
        const auto json_keyfrm = json_id_keyfrm.value();

        register_graph(id, json_keyfrm);
    }

    // Step 5. Register association between keyframs and 3D points
    spdlog::info("registering keyframe-landmark association");
    for (const auto& json_id_keyfrm : json_keyfrms.items()) {
        const auto id = std::stoi(json_id_keyfrm.key());
        assert(0 <= id);
        const auto json_keyfrm = json_id_keyfrm.value();

        register_association(id, json_keyfrm);
    }

    // Step 6. Update graph
    spdlog::info("updating covisibility graph");
    for (const auto& json_id_keyfrm : json_keyfrms.items()) {
        const auto id = std::stoi(json_id_keyfrm.key());
        assert(0 <= id);

        assert(keyframes_.count(id));
        auto keyfrm = keyframes_.at(id);

        keyfrm->graph_node_->update_connections();
        keyfrm->graph_node_->update_covisibility_orders();
    }

    // Step 7. Update geometry
    spdlog::info("updating landmark geometry");
    for (const auto& json_id_landmark : json_landmarks.items()) {
        const auto id = std::stoi(json_id_landmark.key());
        assert(0 <= id);

        assert(landmarks_.count(id));
        auto lm = landmarks_.at(id);

        if (!lm->get_ref_keyframe()) {
            continue;
        }
        lm->update_normal_and_depth();
        lm->compute_descriptor();
    }
}

void map_database::register_keyframe(camera_database* cam_db, bow_vocabulary* bow_vocab, bow_database* bow_db,
                                     const unsigned int id, const nlohmann::json& json_keyfrm) {
    // Metadata
    const auto src_frm_id = json_keyfrm.at("src_frm_id").get<unsigned int>();
    const auto timestamp = json_keyfrm.at("ts").get<double>();
    const auto camera_name = json_keyfrm.at("cam").get<std::string>();
    const auto camera = cam_db->get_camera(camera_name);
    const auto depth_thr = json_keyfrm.at("depth_thr").get<float>();

    // Pose information
    const Mat33_t rot_cw = convert_json_to_rotation(json_keyfrm.at("rot_cw"));
    const Vec3_t trans_cw = convert_json_to_translation(json_keyfrm.at("trans_cw"));
    const auto cam_pose_cw = util::converter::to_eigen_cam_pose(rot_cw, trans_cw);

    // Keypoints information
    const auto num_keypts = json_keyfrm.at("n_keypts").get<unsigned int>();
    // keypts
    const auto json_keypts = json_keyfrm.at("keypts");
    const auto keypts = convert_json_to_keypoints(json_keypts);
    assert(keypts.size() == num_keypts);
    // undist_keypts
    const auto json_undist_keypts = json_keyfrm.at("undists");
    const auto undist_keypts = convert_json_to_undistorted(json_undist_keypts);
    assert(undist_keypts.size() == num_keypts);
    // bearings
    auto bearings = eigen_alloc_vector<Vec3_t>(num_keypts);
    assert(bearings.size() == num_keypts);
    camera->convert_keypoints_to_bearings(undist_keypts, bearings);
    // stereo_x_right
    const auto stereo_x_right = json_keyfrm.at("x_rights").get<std::vector<float>>();
    assert(stereo_x_right.size() == num_keypts);
    // depths
    const auto depths = json_keyfrm.at("depths").get<std::vector<float>>();
    assert(depths.size() == num_keypts);
    // descriptors
    const auto json_descriptors = json_keyfrm.at("descs");
    const auto descriptors = convert_json_to_descriptors(json_descriptors);
    assert(descriptors.rows == static_cast<int>(num_keypts));

    // Scale information in ORB
    const auto num_scale_levels = json_keyfrm.at("n_scale_levels").get<unsigned int>();
    const auto scale_factor = json_keyfrm.at("scale_factor").get<float>();

    // Construct a new object
    auto keyfrm = new data::keyframe(id, src_frm_id, timestamp, cam_pose_cw, camera, depth_thr,
                                     num_keypts, keypts, undist_keypts, bearings, stereo_x_right, depths, descriptors,
                                     num_scale_levels, scale_factor, bow_vocab, bow_db, this);

    // Append to map database
    assert(!keyframes_.count(id));
    keyframes_[keyfrm->id_] = keyfrm;
    if (keyfrm->id_ > max_keyfrm_id_) {
        max_keyfrm_id_ = keyfrm->id_;
    }
    if (id == 0) {
        origin_keyfrm_ = keyfrm;
    }
}

void map_database::register_landmark(const unsigned int id, const nlohmann::json& json_landmark) {
    const unsigned int first_keyfrm_id = json_landmark.at("1st_keyfrm").get<int>();
    const auto pos_w = Vec3_t(json_landmark.at("pos_w").get<std::vector<Vec3_t::value_type>>().data());
    const unsigned int ref_keyfrm_id = json_landmark.at("ref_keyfrm").get<int>();

    data::keyframe* ref_keyfrm = nullptr;
    if (!keyframes_.count(ref_keyfrm_id)) {
        spdlog::debug("Invalid ref_keyfrm_id {} for this landmark {}", ref_keyfrm_id, id);
    }
    else {
        ref_keyfrm = keyframes_.at(ref_keyfrm_id);
    }

    const auto num_visible = json_landmark.at("n_vis").get<unsigned int>();
    const auto num_found = json_landmark.at("n_fnd").get<unsigned int>();

    auto lm = new data::landmark(id, first_keyfrm_id, pos_w, ref_keyfrm,
                                 num_visible, num_found, this);
    assert(!landmarks_.count(id));
    landmarks_[lm->id_] = lm;
}

void map_database::register_graph(const unsigned int id, const nlohmann::json& json_keyfrm) {
    // Graph information
    const auto spanning_parent_id = json_keyfrm.at("span_parent").get<int>();
    const auto spanning_children_ids = json_keyfrm.at("span_children").get<std::vector<int>>();
    const auto loop_edge_ids = json_keyfrm.at("loop_edges").get<std::vector<int>>();

    assert(keyframes_.count(id));
    assert(spanning_parent_id == -1 || keyframes_.count(spanning_parent_id));
    keyframes_.at(id)->graph_node_->set_spanning_parent((spanning_parent_id == -1) ? nullptr : keyframes_.at(spanning_parent_id));
    for (const auto spanning_child_id : spanning_children_ids) {
        assert(keyframes_.count(spanning_child_id));
        keyframes_.at(id)->graph_node_->add_spanning_child(keyframes_.at(spanning_child_id));
    }
    for (const auto loop_edge_id : loop_edge_ids) {
        assert(keyframes_.count(loop_edge_id));
        keyframes_.at(id)->graph_node_->add_loop_edge(keyframes_.at(loop_edge_id));
    }
}

void map_database::register_association(const unsigned int keyfrm_id, const nlohmann::json& json_keyfrm) {
    // Key points information
    const auto num_keypts = json_keyfrm.at("n_keypts").get<unsigned int>();
    const auto landmark_ids = json_keyfrm.at("lm_ids").get<std::vector<int>>();
    assert(landmark_ids.size() == num_keypts);

    assert(keyframes_.count(keyfrm_id));
    auto keyfrm = keyframes_.at(keyfrm_id);
    for (unsigned int idx = 0; idx < num_keypts; ++idx) {
        const auto lm_id = landmark_ids.at(idx);
        if (lm_id < 0) {
            continue;
        }
        if (!landmarks_.count(lm_id)) {
            spdlog::warn("landmark {}: not found in the database", lm_id);
            continue;
        }

        auto lm = landmarks_.at(lm_id);
        keyfrm->add_landmark(lm, idx);
        lm->add_observation(keyfrm, idx);
    }
}

void map_database::to_json(nlohmann::json& json_keyfrms, nlohmann::json& json_landmarks) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    // Save each keyframe as json
    spdlog::info("encoding {} keyframes to store", keyframes_.size());
    std::map<std::string, nlohmann::json> keyfrms;
    for (const auto id_keyfrm : keyframes_) {
        const auto id = id_keyfrm.first;
        const auto keyfrm = id_keyfrm.second;
        assert(keyfrm);
        assert(id == keyfrm->id_);
        assert(!keyfrm->will_be_erased());
        keyfrm->graph_node_->update_connections();
        assert(!keyfrms.count(std::to_string(id)));
        keyfrms[std::to_string(id)] = keyfrm->to_json();
    }
    json_keyfrms = keyfrms;

    // Save each 3D point as json
    spdlog::info("encoding {} landmarks to store", landmarks_.size());
    std::map<std::string, nlohmann::json> landmarks;
    for (const auto id_lm : landmarks_) {
        const auto id = id_lm.first;
        const auto lm = id_lm.second;
        assert(lm);
        assert(id == lm->id_);
        assert(!lm->will_be_erased());
        lm->update_normal_and_depth();
        assert(!landmarks.count(std::to_string(id)));
        landmarks[std::to_string(id)] = lm->to_json();
    }
    json_landmarks = landmarks;
}

void map_database::delete_from_grid(const landmark* lm) {
    std::lock_guard<std::mutex> lock(mtx_mapgrid_access_);
    Vec3_t pos_w = lm->get_pos_in_world();
    landmark_grid_[std::pair<int, int>(pos_w[0] / grid_size_, pos_w[1] / grid_size_)].erase(lm->id_);
}

void map_database::delete_from_grid(const unsigned int id, const int grid_x, const int grid_y) {
    std::lock_guard<std::mutex> lock(mtx_mapgrid_access_);
    landmark_grid_[std::pair<int, int>(grid_x, grid_y)].erase(id);
}

void map_database::insert_into_grid(landmark* lm, int grid_x, int grid_y) {
    std::lock_guard<std::mutex> lock(mtx_mapgrid_access_);
    landmark_grid_[std::pair<int, int>(grid_x, grid_y)][lm->id_] = lm;
}

std::shared_ptr<data::keyframe> map_database::create_virtual_keyfrm(data::frame& frame, data::map_database* map_db,
                                                                    bow_database* bow_db,
                                                                    data::bow_vocabulary* bow_vocab,
                                                                    std::vector<data::landmark*>& nearby_landmarks) {
    std::vector<cv::KeyPoint> keypts, undist_keypts;
    size_t num_keypts = nearby_landmarks.size();
    std::vector<float> stereo_x_right(num_keypts, 0), depths(num_keypts, 0);
    std::vector<data::landmark*> lm_vec(num_keypts, nullptr);
    cv::Mat descriptors(num_keypts, 32, CV_8U);
    openvslam::eigen_alloc_vector<Eigen::Vector3d> bearings;
    bearings.resize(num_keypts);
    auto camera_pose = frame.get_cam_pose();
    unsigned int idx = 0;

    for (auto lm : nearby_landmarks) {
        lm->get_descriptor().copyTo(descriptors.row(idx));
        Eigen::Vector3d lm_in_camera = camera_pose.block(0, 0, 3, 3) * lm->get_pos_in_world() + camera_pose.block(0, 3, 3, 1);

        if (lm_in_camera[2] <= 0)
            lm_vec[idx] = nullptr;
        else {
            lm_vec[idx] = lm;

            lm_in_camera = lm_in_camera / lm_in_camera[2];

            bearings[idx] = lm_in_camera / lm_in_camera.head(2).norm();
        }
        idx++;
    }
    static const unsigned int num_scale_levels = frame.num_scale_levels_;
    static const float scale_factor = frame.scale_factor_;

    std::shared_ptr<data::keyframe> virtual_keyframe = std::make_shared<data::keyframe>(
        static_cast<unsigned int>(-1), static_cast<unsigned int>(-1), frame.timestamp_, camera_pose, frame.camera_,
        frame.depth_thr_, num_keypts, keypts, undist_keypts, bearings, stereo_x_right, depths, descriptors,
        num_scale_levels, scale_factor, bow_vocab, bow_db, map_db);

    for (unsigned int i = 0; i < num_keypts; i++) {
        if (lm_vec[i]) {
            virtual_keyframe->add_landmark(lm_vec[i], i);
        }
    }
    return virtual_keyframe;
}

std::shared_ptr<data::keyframe> map_database::create_virtual_keyfrm(data::keyframe* keyfrm, data::map_database* map_db,
                                                                    bow_database* bow_db,
                                                                    data::bow_vocabulary* bow_vocab,
                                                                    std::vector<data::landmark*>& nearby_landmarks) {
    std::vector<cv::KeyPoint> keypts, undist_keypts;
    size_t num_keypts = nearby_landmarks.size();
    std::vector<float> stereo_x_right(num_keypts, 0), depths(num_keypts, 0);
    std::vector<data::landmark*> lm_vec(num_keypts, nullptr);
    cv::Mat descriptors(num_keypts, 32, CV_8U);
    openvslam::eigen_alloc_vector<Eigen::Vector3d> bearings;
    bearings.resize(num_keypts);
    auto camera_pose = keyfrm->get_cam_pose();
    unsigned int idx = 0;

    for (auto lm : nearby_landmarks) {
        lm->get_descriptor().copyTo(descriptors.row(idx));
        Eigen::Vector3d lm_in_camera = camera_pose.block(0, 0, 3, 3) * lm->get_pos_in_world() + camera_pose.block(0, 3, 3, 1);

        if (lm_in_camera[2] <= 0)
            lm_vec[idx] = nullptr;
        else {
            lm_vec[idx] = lm;

            lm_in_camera = lm_in_camera / lm_in_camera[2];

            bearings[idx] = lm_in_camera / lm_in_camera.head(2).norm();
        }
        idx++;
    }
    static const unsigned int num_scale_levels = keyfrm->num_scale_levels_;
    static const float scale_factor = keyfrm->scale_factor_;

    std::shared_ptr<data::keyframe> virtual_keyframe = std::make_shared<data::keyframe>(
        static_cast<unsigned int>(-1), static_cast<unsigned int>(-1), keyfrm->timestamp_, camera_pose, keyfrm->camera_,
        keyfrm->depth_thr_, num_keypts, keypts, undist_keypts, bearings, stereo_x_right, depths, descriptors,
        num_scale_levels, scale_factor, bow_vocab, bow_db, map_db);

    for (unsigned int i = 0; i < num_keypts; i++) {
        if (lm_vec[i]) {
            virtual_keyframe->add_landmark(lm_vec[i], i);
        }
    }
    return virtual_keyframe;
}

std::vector<landmark*> map_database::get_landmarks_in_frustum(Eigen::Matrix4d curr_pose, camera::base* curr_camera,
                                                              double near, double far, double back) {
    camera::image_bounds bounds = curr_camera->compute_image_bounds();
    const std::vector<cv::KeyPoint> pixels{
        cv::KeyPoint(bounds.min_x_, bounds.min_y_, 1.0),                  // left top
        cv::KeyPoint(bounds.max_x_, bounds.min_y_, 1.0),                  // right top
        cv::KeyPoint(bounds.min_x_, bounds.max_y_, 1.0),                  // left bottom
        cv::KeyPoint(bounds.max_x_, bounds.max_y_, 1.0),                  // right bottom
        /*cv::KeyPoint(bounds.max_x_ * 0.5, bounds.max_y_ * 0.5, 1.0)*/}; // middle - not needed
    eigen_alloc_vector<Vec3_t> bearings;
    curr_camera->convert_keypoints_to_bearings(pixels, bearings);

    // Calculate the world coorindates of the near and far point of each bearing,
    // find out their boundaries and convex hull
    std::vector<cv::Point2f> points2d, hull;
    std::vector<cv::Point3f> point_3d;
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::min();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::min();
    auto update_boundary = [&min_x, &max_x, &min_y, &max_y](double x, double y) {
        if (x < min_x)
            min_x = x;
        if (x > max_x)
            max_x = x;
        if (y < min_y)
            min_y = y;
        if (y > max_y)
            max_y = y;
    };

    // obtain inverse pose
    const Eigen::Matrix3d rot_cw = curr_pose.block<3, 3>(0, 0);
    const Eigen::Vector3d trans_cw = curr_pose.block<3, 1>(0, 3);
    const Eigen::Matrix3d rot_wc = rot_cw.transpose();
    Eigen::Vector3d cam_center = -rot_wc * trans_cw;

    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Twc.block<3, 3>(0, 0) = rot_wc;
    Twc.block<3, 1>(0, 3) = cam_center + rot_wc * Eigen::Vector3d(0, 0, -back);

    // const Eigen::Matrix4d Twc = camera_frame->get_cam_pose_inv(); // T(world, camera)
    Eigen::Vector4d tcp; // t(camera, point)
    tcp << 0, 0, 0, 1;
    for (const auto& bearing : bearings) {
        tcp.block<3, 1>(0, 0) = bearing * near;
        Eigen::Vector4d twp = Twc * tcp;
        points2d.emplace_back(twp(0), twp(1));
        point_3d.emplace_back(twp(0), twp(1), twp(2));
        update_boundary(twp(0), twp(1));
        tcp.block<3, 1>(0, 0) = bearing * far;
        twp = Twc * tcp;
        points2d.emplace_back(twp(0), twp(1));
        point_3d.emplace_back(twp(0), twp(1), twp(2));
        update_boundary(twp(0), twp(1));
    }
    // nice formulation!!
    cv::convexHull(points2d, hull);
    std::vector<cv::Point2f> point_2d;
    std::vector<cv::Point2f> convex_hull;
    for (auto p2d : points2d)
        point_2d.push_back(p2d);
    for (auto ch : hull)
        convex_hull.push_back(ch);

    // Iterate all the grid cells within the boundary
    int min_grid_x = std::floor(min_x / grid_size_);
    int min_grid_y = std::floor(min_y / grid_size_);
    int max_grid_x = std::ceil(max_x / grid_size_);
    int max_grid_y = std::ceil(max_y / grid_size_);

    // prevent occlusion
    std::unordered_map<std::pair<int, int>, std::vector<std::pair<data::landmark*, float>>, pair_hash> occlusion_judge;
    // every 3*3 pixel squares can not have over 5 points
    auto pose_can_observe = [&curr_pose, &curr_camera, &occlusion_judge](data::landmark* lm, float x_expand, float y_expand) {
        const Vec3_t pos_w = lm->get_pos_in_world();
        Vec2_t reproj;
        float right_x;
        bool in_image = curr_camera->reproject_to_image(curr_pose.block(0, 0, 3, 3), curr_pose.block(0, 3, 3, 1), pos_w, reproj,
                                               right_x, x_expand, y_expand);
        if (!in_image)
            return false;
        Vec3_t cam_pose = curr_pose.block(0, 0, 3, 3) * pos_w + curr_pose.block(0, 3, 3, 1);
        size_t max_loc;
        float max_depth = 0;
        std::pair<int, int> max_proj;
        int number = 0;
        for (int i=-1; i<=1; i++) {
            for (int j=-1; j<=1; j++) {
                std::pair<int, int> proj = std::make_pair(static_cast<int> (reproj(0)+i), static_cast<int> (reproj(1)+j));
                if (occlusion_judge.find(proj) != occlusion_judge.end()) {
                    std::vector<std::pair<data::landmark*, float>> curr_lms = occlusion_judge[proj];
                    for (size_t idx=0; idx<curr_lms.size(); idx++) {
                        number++;
                        if (curr_lms[idx].second > max_depth) {
                            max_depth = curr_lms[idx].second;
                            max_loc = idx;
                            max_proj = proj;
                        }
                    }
                }
            }
        }

        if (number >= 1) {
            if (cam_pose(2) >= max_depth)
                return false;
            else {
                occlusion_judge[max_proj].erase(occlusion_judge[max_proj].begin() + max_loc);
                if (occlusion_judge[max_proj].size() == 0)
                    occlusion_judge.erase(max_proj);
            }
        }
        std::pair<int, int> proj = std::make_pair(static_cast<int> (reproj(0)), static_cast<int> (reproj(1)));
        if (occlusion_judge.find(proj) == occlusion_judge.end()) {
            occlusion_judge[proj] = std::vector<std::pair<data::landmark*, float>>();
        }
        occlusion_judge[proj].push_back(std::make_pair(lm, cam_pose(2)));
        return true;
    };

    // auto pose_can_observe = [&curr_pose, &curr_camera](data::landmark* lm, float x_expand, float y_expand) {
    //     const Vec3_t pos_w = lm->get_pos_in_world();
    //     Vec2_t reproj;
    //     float right_x;
    //     return curr_camera->reproject_to_image(curr_pose.block(0, 0, 3, 3), curr_pose.block(0, 3, 3, 1), pos_w, reproj,
    //                                            right_x, x_expand, y_expand);
    // };

    std::vector<data::landmark*> landmarks;
    for (int x = min_grid_x; x <= max_grid_x; ++x) {
        for (int y = min_grid_y; y <= max_grid_y; ++y) {
            if (pointPolygonTest(hull, cv::Point2f(x * grid_size_, y * grid_size_), false) < 0)
                continue; // outside the hull

            gridxy_index_pair cell(x, y);
            if (landmark_grid_.find(cell) == landmark_grid_.end())
                continue;
            const auto& landmarks_in_grid = landmark_grid_[cell];

            for (auto& item : landmarks_in_grid) {
                // Ignore invalid landmarks and landmarks on other maps
                if (!item.second)
                    continue;

                // Ignore landmarks outside of the current camera view (with a tolerance of 10 pixels)
                if (!pose_can_observe(item.second, 10, 10))
                    continue;

                landmarks.push_back(item.second);
            }
        }
    }
    // return landmarks;

    std::vector<data::landmark*> occlusion_landmarks;
    for (auto pixel:occlusion_judge) {
        std::vector<std::pair<data::landmark*, float>> lm_vector = pixel.second;
        for (auto lm:lm_vector)
            occlusion_landmarks.push_back(lm.first);
    }
    return occlusion_landmarks;
    
}

std::vector<landmark*> map_database::get_landmarks_in_covisibility(data::keyframe* keyfrm, int top_number) {
    std::vector<data::keyframe*> covisi_kfs = keyfrm->graph_node_->get_top_n_covisibilities(top_number);
    covisi_kfs.push_back(keyfrm);

    std::vector<data::landmark*> covis_landmarks;
    std::unordered_set<unsigned int> landmarks_ids;
    for (auto kf : covisi_kfs) {
        std::vector<data::landmark*> lms = kf->get_landmarks();
        for (auto lm : lms) {
            if (!lm)
                continue;
            if (landmarks_ids.find(lm->id_) != landmarks_ids.end())
                continue;
            covis_landmarks.push_back(lm);
            landmarks_ids.emplace(lm->id_);
        }
    }

    return covis_landmarks;
}

} // namespace data
} // namespace openvslam
