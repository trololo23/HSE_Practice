#include <pybind11/pybind11.h>

#include "iostream"
#include "cassert"
#include "omp.h"
#include "set"
#include "unordered_map"
#include "vector"
#include "pybind11/stl.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

bool check_size(std::vector<std::vector<int32_t>> &image1, std::vector<std::vector<int32_t>> &image2) {

    if (image1.size() != image2.size() || image1.empty()) {
        return false;
    }

    int32_t width = image1[0].size();

    #pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image1.size(); ++i) {
        if (image1[i].size() != width || image2[i].size() != width) {
            return false;
        }
    }
    return true;
}

int32_t get_vertex_id(std::vector<std::vector<int32_t>> &image, int32_t x, int32_t y) {
    return x * image[0].size() + y;
}

int32_t get_x_in_image(std::vector<std::vector<int32_t>> &image, int32_t vertex_id) {
    return vertex_id / image[0].size();
}

int32_t get_y_in_image(std::vector<std::vector<int32_t>> &image, int32_t vertex_id) {
    return vertex_id % image[0].size();
}

void
get_neighbours_4_connectivity(std::vector<std::vector<int32_t>> &image, std::vector<std::vector<int32_t>> &image_graph,
                              int32_t x, int32_t y) {
    int int_x = static_cast<int>(x);
    int int_y = static_cast<int>(y);

    #pragma omp parallel for num_threads(4) collapse(2)
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (abs(i + j) == 1) {
                int new_x = int_x + i;
                int new_y = int_y + j;
                if (new_x >= 0 && new_y >= 0 && new_x < image.size() && new_y < image[0].size()) {
                    image_graph[get_vertex_id(image, x, y)].emplace_back(get_vertex_id(image, new_x, new_y));
                }
            }
        }
    }
}

void get_neighbours_diagonal_connectivity(std::vector<std::vector<int32_t>> &image,
                                          std::vector<std::vector<int32_t>> &image_graph, int32_t x, int32_t y) {
    int int_x = static_cast<int>(x);
    int int_y = static_cast<int>(y);

    #pragma omp parallel for num_threads(4) collapse(2)
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (abs(i) == 1 && abs(j) == 1) {
                int new_x = int_x + i;
                int new_y = int_y + j;
                if (new_x >= 0 && new_y >= 0 && new_x < image.size() && new_y < image[0].size()) {
                    image_graph[get_vertex_id(image, x, y)].emplace_back(get_vertex_id(image, new_x, new_y));
                }
            }
        }
    }
}

void
get_neighbours_8_connectivity(std::vector<std::vector<int32_t>> &image, std::vector<std::vector<int32_t>> &image_graph,
                              int32_t x, int32_t y) {
    get_neighbours_4_connectivity(image, image_graph, x, y);
    get_neighbours_diagonal_connectivity(image, image_graph, x, y);
}

bool
is_border(std::vector<std::vector<int32_t>> &image, std::vector<std::vector<int32_t>> &image_graph, int32_t vertex,
          bool black) {
    #pragma omp parallel for num_threads(4)
    for (auto i: image_graph[vertex]) {
        int32_t x = get_x_in_image(image, i);
        int32_t y = get_y_in_image(image, i);
        if (!image[x][y] == black) {
            return true;
        }
    }
    return false;
}

void build_graph(std::vector<std::vector<int32_t>> &image, std::vector<std::vector<int32_t>> &image_graph,
                 std::vector<std::vector<int32_t>> &transformed_image, std::string &connectivity_type, bool black) {
    image_graph.resize(image[0].size() * image.size());
    #pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = 0; i < image.size(); ++i) {
        for (int32_t j = 0; j < image[i].size(); ++j) {
            if (connectivity_type == "4-connectivity") {
                get_neighbours_4_connectivity(image, image_graph, i, j);
            } else if (connectivity_type == "8-connectivity") {
                get_neighbours_8_connectivity(image, image_graph, i, j);
            } else if (connectivity_type == "diagonal-connectivity") {
                get_neighbours_diagonal_connectivity(image, image_graph, i, j);
            }
            transformed_image[i][j] = 2 * std::max(image.size(), image[0].size());
        }
    }
}

void update_distances(std::vector<int32_t> &border, std::vector<std::vector<int32_t>> &image_graph,
                      std::vector<std::vector<int32_t>> &transformed_image) {

    int32_t bucket_size = image_graph.size() / 10 + 1;
    int32_t left_bucket = 0;

    std::vector<std::set<std::pair<int32_t, int32_t>>> buckets(image_graph.size() / bucket_size + 1);

    std::set<std::pair<int32_t, int32_t>> priority_queue;
    #pragma omp parallel for num_threads(4)
    for (auto &i: border) {
        buckets[0].insert({0, i});
    }

    while (left_bucket < buckets.size()) {
        int32_t current_distance = buckets[left_bucket].begin()->first;
        int32_t current_vertex = buckets[left_bucket].begin()->second;

        buckets[left_bucket].erase(buckets[left_bucket].begin());
        #pragma omp parallel for num_threads(4)
        for (auto go_to: image_graph[current_vertex]) {
            int32_t current_x = get_x_in_image(transformed_image, go_to);
            int32_t current_y = get_y_in_image(transformed_image, go_to);
            if (transformed_image[current_x][current_y] > current_distance + 1) {

                int32_t new_bucket_id = (current_distance + 1) / bucket_size;
                int32_t previous_bucket_id = (transformed_image[current_x][current_y]) / bucket_size;
                transformed_image[current_x][current_y] = current_distance + 1;

                buckets[previous_bucket_id].erase({transformed_image[current_x][current_y], go_to});
                buckets[new_bucket_id].insert({current_distance + 1, go_to});
            }
        }
        #pragma omp parallel for num_threads(4)
        for (int32_t i = left_bucket; i < buckets.size(); ++i) {
            if (!buckets[i].empty()) {
                break;
            }
            left_bucket++;
        }
    }

    for (int i = 0; i < transformed_image.size(); ++i) {
        for (int j = 0; j < transformed_image[0].size(); ++j) {
            std::cout << transformed_image[i][j] << " ";
        }
        std::cout << "\n";
    }
}

void
euclidean_transformation(std::vector<std::vector<int32_t>> &image, std::vector<std::vector<int32_t>> &transformed_image,
                         std::string &connectivity_type, bool black = true) {
    std::vector<std::vector<int32_t>> image_graph;
    build_graph(image, image_graph, transformed_image, connectivity_type, black);

    std::vector<int32_t> border;
    #pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        int32_t x = get_x_in_image(image, i);
        int32_t y = get_y_in_image(image, i);
        if (image[x][y] == black) {
            transformed_image[x][y] = 0;
            if (is_border(image, image_graph, i, black)) {
                border.push_back(i);
            }
        }
    }
    update_distances(border, image_graph, transformed_image);
}

void signed_euclidean_distance(std::vector<std::vector<int32_t>> &image,
                               std::vector<std::vector<int32_t>> &transformed_image,
                               std::string &connectivity_type) {
    std::vector<std::vector<int32_t>> transformed_image_copy = transformed_image;
    euclidean_transformation(image, transformed_image, connectivity_type, true);
    euclidean_transformation(image, transformed_image_copy, connectivity_type, false);
    #pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = 0; i < image.size(); ++i) {
        for (int32_t j = 0; j < image[0].size(); ++j) {
            transformed_image[i][j] -= transformed_image_copy[i][j];
        }
    }
}

void make_transformation(std::vector<std::vector<int32_t>> &image, std::vector<std::vector<int32_t>> &transformed_image,
                         std::string &transformation_type, std::string &connectivity_type) {
    assert(check_size(image, transformed_image));

    if (transformation_type == "euclidean") {
        euclidean_transformation(image, transformed_image, connectivity_type);
    } else if (transformation_type == "signed_euclidean") {
        signed_euclidean_distance(image, transformed_image, connectivity_type);
    } else {
        std::cerr << "non defined transformation type";
    }
}

namespace py = pybind11;

PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("check_size", &check_size, R"pbdoc(
    )pbdoc");

    m.def("get_vertex_id", &get_vertex_id, R"pbdoc(
    )pbdoc");

    m.def("get_x_in_image", &get_x_in_image, R"pbdoc(
    )pbdoc");

    m.def("get_y_in_image", &get_y_in_image, R"pbdoc(
    )pbdoc");

    m.def("get_neighbours_4_connectivity", &get_neighbours_4_connectivity, R"pbdoc(
    )pbdoc");

    m.def("get_neighbours_diagonal_connectivity", &get_neighbours_diagonal_connectivity, R"pbdoc(
    )pbdoc");

    m.def("get_neighbours_8_connectivity", &get_neighbours_8_connectivity, R"pbdoc(
    )pbdoc");

    m.def("is_border", &is_border, R"pbdoc(
    )pbdoc");

    m.def("build_graph", &build_graph, R"pbdoc(
    )pbdoc");

    m.def("update_distances", &update_distances, R"pbdoc(
    )pbdoc");

    m.def("euclidean_transformation", &euclidean_transformation, R"pbdoc(
    )pbdoc");
    
    m.def("signed_euclidean_distance", &get_vertex_id, R"pbdoc(
    )pbdoc");

    m.def("make_transformation", &make_transformation, R"pbdoc(
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
