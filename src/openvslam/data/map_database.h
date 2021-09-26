#ifndef OPENVSLAM_DATA_MAP_DATABASE_H
#define OPENVSLAM_DATA_MAP_DATABASE_H

#include "openvslam/data/bow_vocabulary.h"
#include "openvslam/data/frame_statistics.h"

#include <mutex>
#include <vector>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

template<typename T>
inline void hash_combine(std::size_t& seed, const T& val) {
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
// auxiliary generic functions to create a hash value using a seed
template<typename T>
inline void hash_val(std::size_t& seed, const T& val) {
    hash_combine(seed, val);
}
template<typename T, typename... Types>
inline void hash_val(std::size_t& seed, const T& val, const Types&... args) {
    hash_combine(seed, val);
    hash_val(seed, args...);
}

template<typename... Types>
inline std::size_t hash_val(const Types&... args) {
    std::size_t seed = 0;
    hash_val(seed, args...);
    return seed;
}
struct pair_hash {
    template<class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return hash_val(p.first, p.second);
    }
};

namespace openvslam {

namespace camera {
class base;
} // namespace camera

typedef std::pair<int, int> gridxy_index_pair;

namespace data {

class frame;
class keyframe;
class landmark;
class camera_database;
class bow_database;

class map_database {
public:
    /**
     * Constructor
     */
    map_database();

    /**
     * Destructor
     */
    ~map_database();

    /**
     * Add keyframe to the database
     * @param keyfrm
     */
    void add_keyframe(keyframe* keyfrm);

    /**
     * Erase keyframe from the database
     * @param keyfrm
     */
    void erase_keyframe(keyframe* keyfrm);

    /**
     * Add landmark to the database
     * @param lm
     */
    void add_landmark(landmark* lm);

    /**
     * Erase landmark from the database
     * @param lm
     */
    void erase_landmark(landmark* lm);

    /**
     * Set local landmarks
     * @param local_lms
     */
    void set_local_landmarks(const std::vector<landmark*>& local_lms);

    /**
     * Get local landmarks
     * @return
     */
    std::vector<landmark*> get_local_landmarks() const;

    /**
     * Get all of the keyframes in the database
     * @return
     */
    std::vector<keyframe*> get_all_keyframes() const;

    /**
     * Get closest keyframes to a given pose
     * @param pose Given pose
     * @param distance_threshold Maximum distance where close keyframes could be found
     * @param angle_threshold Maximum angle between given pose and close keyframes
     * @return Vector closest keyframes
     */
    std::vector<keyframe*> get_close_keyframes(const Mat44_t& pose,
                                               const double distance_threshold,
                                               const double angle_threshold) const;

    /**
     * Get the number of keyframes
     * @return
     */
    unsigned get_num_keyframes() const;

    /**
     * Get all of the landmarks in the database
     * @return
     */
    std::vector<landmark*> get_all_landmarks() const;

    /**
     * Get the number of landmarks
     * @return
     */
    unsigned int get_num_landmarks() const;

    /**
     * Get the maximum keyframe ID
     * @return
     */
    unsigned int get_max_keyframe_id() const;

    /**
     * Update frame statistics
     * @param frm
     * @param is_lost
     */
    void update_frame_statistics(const data::frame& frm, const bool is_lost) {
        std::lock_guard<std::mutex> lock(mtx_map_access_);
        frm_stats_.update_frame_statistics(frm, is_lost);
    }

    /**
     * Replace a keyframe which will be erased in frame statistics
     * @param old_keyfrm
     * @param new_keyfrm
     */
    void replace_reference_keyframe(data::keyframe* old_keyfrm, data::keyframe* new_keyfrm) {
        std::lock_guard<std::mutex> lock(mtx_map_access_);
        frm_stats_.replace_reference_keyframe(old_keyfrm, new_keyfrm);
    }

    /**
     * Get frame statistics
     * @return
     */
    frame_statistics get_frame_statistics() const {
        std::lock_guard<std::mutex> lock(mtx_map_access_);
        return frm_stats_;
    }

    /**
     * Clear the database
     */
    void clear();

    /**
     * Load keyframes and landmarks from JSON
     * @param cam_db
     * @param bow_vocab
     * @param bow_db
     * @param json_keyfrms
     * @param json_landmarks
     */
    void from_json(camera_database* cam_db, bow_vocabulary* bow_vocab, bow_database* bow_db,
                   const nlohmann::json& json_keyfrms, const nlohmann::json& json_landmarks);

    /**
     * Dump keyframes and landmarks as JSON
     * @param json_keyfrms
     * @param json_landmarks
     */
    void to_json(nlohmann::json& json_keyfrms, nlohmann::json& json_landmarks);

    //! origin keyframe
    keyframe* origin_keyfrm_ = nullptr;

    //! mutex for locking ALL access to the database
    //! (NOTE: cannot used in map_database class)
    static std::mutex mtx_database_;

    void delete_from_grid(const landmark* lm);

    void delete_from_grid(const unsigned int id, const int grid_x, const int grid_y);

    void insert_into_grid(landmark* lm, int grid_x, int grid_y);

    // T is keyframe or frame

    std::shared_ptr<data::keyframe> create_virtual_keyfrm(data::keyframe* keyfrm, data::map_database* map_db,
                                                          bow_database* bow_db,
                                                          data::bow_vocabulary* bow_vocab,
                                                          std::vector<data::landmark*>& nearby_landmarks);

    std::shared_ptr<data::keyframe> create_virtual_keyfrm(data::frame& frame, data::map_database* map_db,
                                                          bow_database* bow_db,
                                                          data::bow_vocabulary* bow_vocab,
                                                          std::vector<data::landmark*>& nearby_landmarks);

    std::vector<landmark*> get_landmarks_in_frustum(Eigen::Matrix4d curr_pose, camera::base* curr_camera,
                                                    double near, double far, double back);

    std::vector<landmark*> get_landmarks_in_covisibility(data::keyframe* keyfrm, int top_number);

    double grid_size_ = 0.1;

private:
    /**
     * Decode JSON and register keyframe information to the map database
     * (NOTE: objects which are not constructed yet will be set as nullptr)
     * @param cam_db
     * @param bow_vocab
     * @param bow_db
     * @param id
     * @param json_keyfrm
     */
    void register_keyframe(camera_database* cam_db, bow_vocabulary* bow_vocab, bow_database* bow_db,
                           const unsigned int id, const nlohmann::json& json_keyfrm);

    /**
     * Decode JSON and register landmark information to the map database
     * (NOTE: objects which are not constructed yet will be set as nullptr)
     * @param id
     * @param json_landmark
     */
    void register_landmark(const unsigned int id, const nlohmann::json& json_landmark);

    /**
     * Decode JSON and register essential graph information
     * (NOTE: keyframe database must be completely constructed before calling this function)
     * @param id
     * @param json_keyfrm
     */
    void register_graph(const unsigned int id, const nlohmann::json& json_keyfrm);

    /**
     * Decode JSON and register keyframe-landmark associations
     * (NOTE: keyframe and landmark database must be completely constructed before calling this function)
     * @param keyfrm_id
     * @param json_keyfrm
     */
    void register_association(const unsigned int keyfrm_id, const nlohmann::json& json_keyfrm);

    //! mutex for mutual exclusion controll between class methods
    mutable std::mutex mtx_map_access_;

    mutable std::mutex mtx_mapgrid_access_;

    //-----------------------------------------
    // keyframe and landmark database

    //! IDs and keyframes
    std::unordered_map<unsigned int, keyframe*> keyframes_;
    //! IDs and landmarks
    std::unordered_map<unsigned int, landmark*> landmarks_;

    //! local landmarks
    std::vector<landmark*> local_landmarks_;

    //! max keyframe ID
    unsigned int max_keyfrm_id_ = 0;

    //-----------------------------------------
    // frame statistics for odometry evaluation

    //! frame statistics
    frame_statistics frm_stats_;

    std::map<gridxy_index_pair, std::unordered_map<unsigned int, landmark*>> landmark_grid_;
};

} // namespace data
} // namespace openvslam

#endif // OPENVSLAM_DATA_MAP_DATABASE_H
