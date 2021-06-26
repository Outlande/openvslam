#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <spdlog/spdlog.h>
#include <iostream>

int main(int argc, char const* argv[]) {
    if (argc != 3) {
        std::cout << "usage: univloc_to_openvslam  <msg_path> <save_path>" << std::endl;
        return -1;
    }

    std::string path = argv[1];
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        spdlog::critical("cannot load the file at {}", path);
        throw std::runtime_error("cannot load the file at " + path);
    }

    spdlog::info("load the MessagePack file of database from {}", path);
    std::vector<uint8_t> msgpack;
    while (true) {
        uint8_t buffer;
        ifs.read(reinterpret_cast<char*>(&buffer), sizeof(uint8_t));
        if (ifs.eof()) {
            break;
        }
        msgpack.push_back(buffer);
    }
    ifs.close();

    // 3. parse into JSON

    auto json = nlohmann::json::from_msgpack(msgpack);

    // for (auto& el : json.items()) {
    //     std::cout << el.key() << std::endl;
    // }

    // std::cout<<"cameras: "<<json.at("cameras")<<std::endl;
    // std::cout<<"frame_next_id: "<<json.at("frame_next_id")<<std::endl;
    // std::cout<<"keyframe_next_id: "<<json.at("keyframe_next_id")<<std::endl;
    // std::cout<<"landmark_next_id: "<<json.at("landmark_next_id")<<std::endl;

    auto cameras = json.at("cameras");
    for (auto id : cameras.items()) {
        auto dicts = cameras.at(id.key());
        dicts.erase("name");
        if (dicts.at("setup_type") == "Monocular_Inertial"){
            dicts.at("setup_type") = "Monocular";
        }
        cameras.at(id.key()) = dicts;
    }
    json.at("cameras") = cameras;


    auto keyframes = json.at("keyframes");
    for (auto id : keyframes.items()) {
        auto dicts = keyframes.at(id.key());
        size_t num_keypts = dicts.at("keypts").size();
        dicts.at("src_frm_id") = std::stoi(dicts.at("src_frm_id").get<std::string>());
        dicts.erase("client_id");
        dicts["depths"] = std::vector<float> (num_keypts, 0);
        dicts.erase("front_and_rear_constraint_keyframe_id");
        dicts.erase("map_id");

        dicts.at("span_parent") = std::stoi(dicts.at("span_parent").get<std::string>());

        auto spanning_children_ids_str = dicts.at("span_children").get<std::vector<std::string>>();
        auto loop_edge_ids_str = dicts.at("loop_edges").get<std::vector<std::string>>();
        auto landmark_ids_str = dicts.at("lm_ids").get<std::vector<std::string>>();

        std::vector<int> spanning_children_ids, loop_edge_ids, landmark_ids;
        for (size_t idx = 0; idx<spanning_children_ids_str.size(); idx++){
            spanning_children_ids.push_back(std::stoi(spanning_children_ids_str[idx]));
        }
        for (size_t idx = 0; idx<loop_edge_ids_str.size(); idx++){
            loop_edge_ids.push_back(std::stoi(loop_edge_ids_str[idx]));
        }
        for (size_t idx = 0; idx<landmark_ids_str.size(); idx++){
            landmark_ids.push_back(std::stoi(landmark_ids_str[idx]));
        }
        dicts.at("span_children") = spanning_children_ids;
        dicts.at("loop_edges") = loop_edge_ids;
        dicts.at("lm_ids") = landmark_ids;

        keyframes.at(id.key()) = dicts;
    }
    json.at("keyframes") = keyframes;


    auto landmarks = json.at("landmarks");
    for (auto id : landmarks.items()) {
        auto dicts = landmarks.at(id.key());
        dicts.at("1st_keyfrm") = std::stoi(dicts.at("1st_keyfrm").get<std::string>());
        dicts.at("ref_keyfrm") = std::stoi(dicts.at("ref_keyfrm").get<std::string>());
        dicts.erase("client_id");
        dicts.erase("map_id");

        landmarks.at(id.key()) = dicts;
    }
    json.at("landmarks") = landmarks;


    std::string save_path = argv[2];
    std::ofstream ofs(save_path, std::ios::out | std::ios::binary);

    if (ofs.is_open()) {
        spdlog::info("save the MessagePack file of database to {}", save_path);
        const auto msgpack = nlohmann::json::to_msgpack(json);
        ofs.write(reinterpret_cast<const char*>(msgpack.data()), msgpack.size() * sizeof(uint8_t));
        ofs.close();
    }
    else {
        spdlog::critical("cannot create a file at {}", save_path);
    }

    // auto ori_kf = json.at("keyframes").at("0");
    // for (auto id : ori_kf.items()) {
    //     std::cout << id.key() << std::endl;
    // }
    // std::cout<<ori_kf.at("cam")<<std::endl;
}