#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include "demo/define.h"
#include "demo/demo.h"
#include "yaml-cpp/yaml.h"

namespace {
std::string toLower(const std::string& input) {
    std::string out = input;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}
}  // namespace

std::vector<FeederSetting> Demo::loadFeederSettingYAML(const std::string& path) {
    const std::map<std::string, FeederType> feeder_type_map = {
        {"CAMERA", FeederType::CAMERA},
        {"IPCAMERA", FeederType::IPCAMERA},
        {"VIDEO", FeederType::VIDEO},
    };

    std::vector<FeederSetting> feeder_settings;

    YAML::Node fs_node = YAML::LoadFile(path);
    for (int i = 0; i < fs_node.size(); i++) {
        FeederSetting fs;
        const std::string feeder_type_str = fs_node[i]["feeder_type"].as<std::string>();
        const auto feeder_type_it = feeder_type_map.find(feeder_type_str);
        if (feeder_type_it == feeder_type_map.end()) {
            throw std::invalid_argument("Unsupported feeder_type in FeederSetting.yaml: " +
                                        feeder_type_str);
        }
        fs.feeder_type = feeder_type_it->second;

        YAML::Node src_path_node = fs_node[i]["src_path"];
        for (int j = 0; j < src_path_node.size(); j++) {
            fs.src_path.push_back(src_path_node[j].as<std::string>());
        }

        feeder_settings.push_back(fs);
    }
    return feeder_settings;
}

std::vector<ModelSetting> Demo::loadModelSettingYAML(const std::string& path) {
    const std::map<std::string, ModelType> model_type_map = {
        {"OBJECT", ModelType::OBJECT},
    };
    const std::map<std::string, InputDataType> input_type_map = {
        {"uint8", InputDataType::UINT8},
        {"float32", InputDataType::FLOAT32},
    };
    const std::map<std::string, mobilint::Cluster> cluster_map = {
        {"Cluster0", mobilint::Cluster::Cluster0},
        {"Cluster1", mobilint::Cluster::Cluster1},
    };
    const std::map<std::string, mobilint::Core> core_map = {
        {"Core0", mobilint::Core::Core0},
        {"Core1", mobilint::Core::Core1},
        {"Core2", mobilint::Core::Core2},
        {"Core3", mobilint::Core::Core3},
    };

    std::vector<ModelSetting> model_settings;

    YAML::Node ms_node = YAML::LoadFile(path);
    for (int i = 0; i < ms_node.size(); i++) {
        ModelSetting ms;
        const std::string model_type_str = ms_node[i]["model_type"].as<std::string>();
        const auto model_type_it = model_type_map.find(model_type_str);
        if (model_type_it == model_type_map.end()) {
            throw std::invalid_argument("Unsupported model_type in ModelSetting.yaml: " +
                                        model_type_str);
        }
        ms.model_type = model_type_it->second;
        ms.mxq_path = ms_node[i]["mxq_path"].as<std::string>();
        ms.dev_no = ms_node[i]["dev_no"].as<int>();

        YAML::Node subconfig_node = ms_node[i]["subconfig"];
        if (subconfig_node && subconfig_node["input"]) {
            const std::string input_str =
                toLower(subconfig_node["input"].as<std::string>());
            const auto input_type_it = input_type_map.find(input_str);
            if (input_type_it == input_type_map.end()) {
                throw std::invalid_argument("Unsupported subconfig.input: " + input_str);
            }
            ms.input_type = input_type_it->second;
        } else {
            ms.input_type = InputDataType::FLOAT32;
        }

        YAML::Node core_id_node = ms_node[i]["core_id"];
        for (int j = 0; j < core_id_node.size(); j++) {
            mobilint::Cluster cluster =
                cluster_map.at(core_id_node[j]["cluster"].as<std::string>());
            mobilint::Core core = core_map.at(core_id_node[j]["core"].as<std::string>());
            ms.core_id.push_back({cluster, core});
        }

        YAML::Node num_core_node = ms_node[i]["num_core"];
        if (num_core_node) {
            ms.num_core = num_core_node.as<int>();
            ms.is_num_core = true;
            if (ms.num_core <= 0) {
                std::cerr << "[WARNING] Model index " << i << ": num_core is "
                          << ms.num_core << ". num_core will be 0" << std::endl;
                ms.num_core = 0;
            }
        } else {
            ms.num_core = ms.core_id.size();
            ms.is_num_core = false;
            if (ms.num_core <= 0) {
                std::cerr << "[WARNING] Model index " << i
                          << ": neither num_core nor core_id is set. num_core will be 1"
                          << std::endl;
                ms.num_core = 1;
                ms.is_num_core = true;
            }
        }
        model_settings.push_back(ms);
    }
    return model_settings;
}

LayoutSetting Demo::loadLayoutSettingYAML(const std::string& path) {
    LayoutSetting layout_setting;

    YAML::Node layout_node = YAML::LoadFile(path);

    YAML::Node image_layout_node = layout_node["image_layout"];
    for (int i = 0; i < image_layout_node.size(); i++) {
        std::string image_path = image_layout_node[i]["path"].as<std::string>();
        int x = image_layout_node[i]["roi"][0].as<int>();
        int y = image_layout_node[i]["roi"][1].as<int>();
        int w = image_layout_node[i]["roi"][2].as<int>();
        int h = image_layout_node[i]["roi"][3].as<int>();
        cv::Mat img = cv::imread(image_path);
        cv::resize(img, img, {w, h});

        ImageLayout image_layout = {img, {x, y, w, h}};
        layout_setting.image_layout.push_back(image_layout);
    }

    YAML::Node worker_layout_node = layout_node["worker_layout"];
    for (int i = 0; i < worker_layout_node.size(); i++) {
        int feeder_index = worker_layout_node[i]["feeder_index"].as<int>();
        int model_index = worker_layout_node[i]["model_index"].as<int>();
        int x = worker_layout_node[i]["roi"][0].as<int>();
        int y = worker_layout_node[i]["roi"][1].as<int>();
        int w = worker_layout_node[i]["roi"][2].as<int>();
        int h = worker_layout_node[i]["roi"][3].as<int>();
        WorkerLayout worker_layout = {feeder_index, model_index, {x, y, w, h}};
        layout_setting.worker_layout.push_back(worker_layout);
    }

    return layout_setting;
}
